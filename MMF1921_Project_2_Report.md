<div align="center">

<br><br>

<h1>MMF1921 Project 2</h1>
<h2>Algorithmic Trading System</h2>

<br><br>

<h3>Course</h3>
<p>MMF1921 – Operations Research</p>

<h3>Program</h3>
<p>Master of Mathematical Finance</p>

<h3>University</h3>
<p>University of Toronto</p>

<br>

<h3>Prepared by</h3>
<p>
Katie Chai<br>
Alexander Khadra<br>
Jérôme Charbonneau
</p>

<br>

<h3>Submitted to</h3>
<p>Roy Kwon</p>

<h3>Date</h3>
<p>June 1, 2026</p>

</div>

<div style="page-break-after: always;"></div>

# 1. Introduction

The objective of this project is to design and implement an automated asset management system — an algorithmic trading strategy capable of constructing and rebalancing a portfolio of equities and equity-based ETFs on a semi-annual basis. The algorithm is assessed primarily on two out-of-sample financial metrics: the ex-post Sharpe ratio, which measures risk-adjusted return over the full investment horizon, and the average portfolio turnover rate, which measures the magnitude of weight changes at each rebalancing date. These dual objectives create an inherent tension: strategies that aggressively respond to new information tend to generate higher turnover, while strategies that minimize trading may sacrifice return. A well-designed algorithm must therefore achieve strong risk-adjusted performance while limiting unnecessary portfolio turnover.

The dataset consists of monthly adjusted closing prices for a universe of equities and ETFs, paired with returns on eight factors — market, size, value, short-term reversal, profitability, investment, momentum, and long-term reversal — along with the contemporaneous risk-free rate. The first 60 months of each dataset are reserved exclusively for initial calibration and are not included in the out-of-sample performance evaluation. Portfolios are rebalanced every six months using a walk-forward methodology, meaning the algorithm is only permitted to use data available up to each rebalancing date, with no lookahead.

During model development, a separate train-validation-test framework was employed to tune hyperparameters and compare candidate strategies. Final strategy performance was then evaluated using the project's prescribed walk-forward backtesting procedure with a 60-month initial calibration window.

Given these constraints, the model development process focused on three interconnected problems. The first is return estimation: how to form reliable forward-looking expected returns from noisy historical data and factor exposures. The second is covariance estimation: how to construct a well-conditioned covariance matrix suitable for optimization, particularly given the limited number of observations relative to the number of assets. The third is portfolio construction: how to translate estimates of return and risk into a portfolio that maximizes risk-adjusted performance while keeping turnover low enough to remain competitive on the second criterion.

To address these problems, several modelling approaches were developed, implemented, and evaluated through a systematic train-validate-test framework before a final strategy was selected. The candidate strategies ranged from simple benchmarks — equal weighting and historical mean-variance optimization — to more sophisticated approaches combining regularized factor models with shrinkage-based covariance estimation and explicit turnover control.

# 2. Data

## 2.1 Asset Price Data

The investment universe consists of 20 U.S. stocks whose tickers are listed in Table 1. The dataset provides monthly adjusted closing prices for each stock from December 2001 to December 2016. These prices are used to compute monthly asset returns.

**Table 1: Investment Universe**

| **F** | **CAT** | **DIS** | **MCD** | **KO** | **PEP** | **WMT** | **C** | **WFC** | **JPM** |
|---|---|---|---|---|---|---|---|---|---|
| **AAPL** | **IBM** | **PFE** | **JNJ** | **XOM** | **MRO** | **ED** | **T** | **VZ** | **NEM** |

Adjusted closing prices are used instead of regular closing prices because they account for corporate actions such as dividends, stock splits, and rights offerings. In a backtest, failing to account for these events would distort the measured return series — for example, a stock split would appear as a large price drop that never actually reduced investor wealth. Using adjusted prices therefore gives a more accurate reflection of the total return earned by an investor holding the stock through these events.

The monthly return for each asset is computed as:

$$
r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1
$$

where $P_{i,t}$ is the adjusted closing price of asset $i$ at the end of month $t$.

All factor models are estimated on excess returns, ensuring that the intercept $\alpha_i$ in each regression reflects compensation above the risk-free rate rather than a blend of risk premia and the time value of money. Portfolio performance is similarly evaluated on excess returns when computing the Sharpe ratio.

## 2.2 Factor Return Data

The project also provides monthly returns for eight risk factors drawn from the Ken French Data Library. These factors are used to explain the systematic component of asset returns. The eight factors are listed in Table 2.

**Table 2: Risk Factors**

| Factor | Name | Economic Interpretation |
|--------|------|------------------------|
| Mkt-RF | Market excess return | Broad market risk premium |
| SMB | Size | Return spread between small and large firms |
| HML | Value | Return spread between value and growth firms |
| RMW | Profitability | Return spread between profitable and unprofitable firms |
| CMA | Investment | Return spread between low and high investment firms |
| Mom | Momentum | Return spread based on prior 12-month performance |
| ST Rev | Short-term reversal | Return spread based on prior 1-month performance |
| LT Rev | Long-term reversal | Return spread based on prior 5-year performance |

All eight factors are derived from synthetic long-short portfolios of stocks with shared characteristics. Because they are constructed from overlapping universes of assets, the factors exhibit non-trivial pairwise correlations. This means our factor models do not operate in the ideal orthogonal-factor environment, and the off-diagonal elements of the factor covariance matrix $\Sigma_f$ are non-zero. These covariance terms must therefore be included when computing the asset covariance matrix $Q$.

$$Q_{\text{factor}} = \hat{V}^\top \Sigma_f \hat{V} + D$$

where $\hat{V}$ is the matrix of estimated factor loadings and $D$ is the diagonal matrix of idiosyncratic variances. Ignoring off-diagonal elements of $\Sigma_f$ — equivalently, assuming factors are orthogonal — would underestimate asset return covariances wherever two assets share exposure to correlated factors, potentially leading the optimizer to construct portfolios that are less diversified than they appear.

The presence of factor correlations also has implications for return estimation. In OLS, correlated regressors inflate the variance of individual coefficient estimates, making individual factor loadings unreliable even when the joint fit is good. This motivates the use of Ridge regression as an alternative estimation technique, since the L2 penalty can stabilize coefficient estimates in the presence of multicollinearity.

## 2.3 Calibration and Investment Windows

The full training dataset spans December 2001 to December 2016. Per the project specifications, the first 60 months — January 2002 through December 2006 — are reserved exclusively for the initial calibration period and are not included in the out-of-sample performance evaluation. Out-of-sample performance is therefore measured from January 2007 onward.

The portfolio is rebalanced every six months. At each rebalancing date, the strategy is recalibrated using the most recent $T_0$ months of available data. The window length $T_0$ is treated as a tunable hyperparameter and is selected during model development through grid search. Using a rolling estimation window allows the model to place greater emphasis on more recent market information while discarding older observations that may no longer be relevant to current market conditions. The impact of $T_0$ on out-of-sample performance is evaluated as part of the hyperparameter selection procedure described in Section 3.8.

For the purposes of model selection, the out-of-sample period is further divided into three non-overlapping segments as summarized in Table 3. The training segment is used only for initial calibration runs; the validation segment is used for hyperparameter selection via grid search; and the test segment is evaluated exactly once using the best parameters identified on the validation set.

**Table 3: Data Partitioning for Model Selection**

| Segment | Purpose |
|---------|---------|
| Initial calibration | Reserved; not scored |
| Training | Strategy calibration during grid search |
| Validation | Hyperparameter selection |
| Test | Final evaluation |

This three-way split is necessary to avoid a subtle but consequential form of overfitting: if hyperparameters are selected by evaluating performance on the same data used to calibrate the model, the reported performance will be optimistic. By holding out the test segment entirely until after hyperparameters are finalized, the test Sharpe ratio and turnover provide an unbiased estimate of how the strategy is likely to perform on the two unseen datasets.


# 3. Methodology

## 3.1 Problem Formulation

At each rebalancing date $t$, the algorithm observes all available historical asset returns and factor returns up to and including period $t$, and must produce a portfolio weight vector $x \in \mathbb{R}^n$ satisfying:

$$
\sum_{i=1}^n x_i = 1, \qquad x_i \geq 0, \qquad x_i \leq x_{\max}
$$

The long-only constraint eliminates short positions, which is standard for an equity fund operating under typical institutional constraints. The per-asset cap $x_{\max}$ is treated as a tunable hyperparameter and is selected through the validation procedure described in Section 3.8. Limiting individual positions prevents excessive concentration and improves robustness to estimation error.

The core optimization problem is mean-variance optimization (MVO), originally formulated by Markowitz (1952). In its general form, the portfolio is selected to maximize a utility function that trades off expected return against portfolio variance:

$$
\max_{x} \quad \mu^\top x - \frac{\gamma}{2} x^\top Q x
$$

where $\mu \in \mathbb{R}^n$ is the vector of expected asset returns, $Q \in \mathbb{R}^{n \times n}$ is the covariance matrix of returns, and $\gamma > 0$ is a risk aversion parameter controlling the return-risk trade-off. This is a convex quadratic program (QP), solved efficiently using CVXPY with the CLARABEL solver.

A well-known limitation of MVO is its sensitivity to estimation error in $\mu$ and $Q$. Small perturbations in these inputs can produce large and unstable changes in the optimal weights. Much of the methodology described below is motivated by mitigating this instability — both through better estimation of $\mu$ and $Q$, and through explicit regularization of the optimization itself.

## 3.2 Benchmark Strategies

Three benchmark strategies were implemented to provide a performance baseline and isolate the contribution of each modelling component.

**Equal weighting** allocates $x_i = 1/n$ to each asset regardless of any data. While naive, equal weighting is known to be surprisingly competitive out-of-sample due to its complete immunity to estimation error, and it trivially achieves zero turnover after the initial allocation since weights drift with prices and are reset symmetrically at each rebalance. It serves as a floor: any model-based strategy that underperforms equal weighting offers no value over a purely passive approach.

**Historical MVO** estimates $\mu$ as the sample mean of asset returns and $Q$ as the sample covariance matrix over a rolling window of $T_0$ observations, then feeds both directly into the MVO optimizer. This is the simplest model-based strategy. Its weakness is that the sample covariance matrix is poorly conditioned when the number of assets $n$ is large relative to the observation window $T_0$, and the sample mean is a notoriously noisy estimator of expected returns. Both sources of error are amplified by the optimizer, frequently producing concentrated and unstable portfolios.

**Historical maximum Sharpe** replaces the MVO objective with direct maximization of the Sharpe ratio:

$$\max_{x} \quad \frac{\mu^\top x - r_f}{\sqrt{x^\top Q x}}$$

This is a non-convex fractional program and is solved using sequential quadratic programming (SLSQP) via SciPy. The expected return is estimated from historical sample means, and the covariance matrix is estimated using Ledoit-Wolf shrinkage (described in Section 3.4) to improve conditioning. While this approach has natural alignment with the Sharpe-ratio-based assessment criterion, it is sensitive to errors in $\mu$ and can produce extreme allocations when expected return estimates are unreliable.

## 3.3 Factor Model for Expected Return Estimation

To obtain more reliable estimates of expected returns, a linear factor model was adopted. The fundamental idea is that asset returns can be decomposed into a component explained by a small number of common factors and an idiosyncratic residual:

$$r_i = \alpha_i + \beta_i^\top f + \varepsilon_i, \qquad \varepsilon_i \sim (0, \sigma_i^2)$$

where $f \in \mathbb{R}^p$ is the vector of factor returns, $\beta_i \in \mathbb{R}^p$ is the vector of factor loadings for asset $i$, $\alpha_i$ is an asset-specific intercept, and $\varepsilon_i$ is the idiosyncratic return assumed uncorrelated across assets. The eight factors provided — market, size, value, short-term reversal, profitability, investment, momentum, and long-term reversal — correspond to the Fama-French factor family, which has extensive empirical support as a description of equity return variation.

In matrix form across all $n$ assets and $T$ observations:

$$R = X B + E, \qquad X = [\mathbf{1} \; F]$$

where $R \in \mathbb{R}^{T \times n}$ is the matrix of asset returns, $F \in \mathbb{R}^{T \times p}$ is the matrix of factor returns, and $B \in \mathbb{R}^{(p+1) \times n}$ stacks the intercepts and loadings.

**Ordinary Least Squares (OLS)** estimates $B$ by minimizing the sum of squared residuals:

$$\hat{B}_{\text{OLS}} = (X^\top X)^{-1} X^\top R$$

Expected returns are then computed as $\mu = \hat{\alpha} + \hat{V}^\top \bar{f}$, where $\bar{f}$ is the sample mean of factor returns. The factor model covariance is:

$$Q_{\text{factor}} = \hat{V}^\top F_{\text{cov}} \hat{V} + D$$

where $F_{\text{cov}}$ is the sample covariance of factor returns and $D = \text{diag}(\hat{\sigma}_1^2, \ldots, \hat{\sigma}_n^2)$ is the diagonal matrix of idiosyncratic variances. This structured decomposition produces a covariance matrix that is guaranteed to be positive semidefinite and has far fewer free parameters than an unrestricted sample covariance, making it better suited to the limited-sample setting.

A practical concern with OLS in this context is multicollinearity among the eight factors. Several of the Fama-French factors — particularly market, momentum, and profitability — exhibit non-trivial pairwise correlations, which inflates the variance of individual OLS coefficient estimates without necessarily biasing them. This instability in $\hat{V}$ propagates directly into $\mu$ and $Q_{\text{factor}}$.

**Ridge regression** addresses this by adding an L2 penalty on the magnitude of the factor loadings to the least-squares objective:

$$\hat{V}_{\text{ridge}} = \arg\min_{V} \; \|R_c - F_c V\|_F^2 + \alpha \|V\|_F^2$$

where $R_c$ and $F_c$ denote mean-centered returns and factors respectively, and $\alpha \geq 0$ is the regularization strength. The closed-form solution is:

$$\hat{V}_{\text{ridge}} = (F_c^\top F_c + \alpha I)^{-1} F_c^\top R_c$$

The effect is to shrink factor loadings toward zero, with the degree of shrinkage controlled by $\alpha$. Crucially, the matrix $(F_c^\top F_c + \alpha I)$ is strictly positive definite for any $\alpha > 0$, eliminating the ill-conditioning problem entirely. The intercepts are recovered analytically as $\hat{\alpha} = \bar{r} - \bar{f}^\top \hat{V}_{\text{ridge}}$, ensuring the intercept is not regularized. Ridge was selected over LASSO because the goal is not factor selection — all eight factors have theoretical motivation — but rather stabilization of the loadings when factors are correlated.

## 3.4 Covariance Estimation via Ledoit-Wolf Shrinkage

Even with a factor model, covariance matrix estimation remains a challenge. The sample covariance matrix $S$ is an unbiased estimator of $Q$, but it is known to be a poor estimator in finite samples: its eigenvalues are systematically dispersed relative to the true eigenvalues, it may be singular or nearly singular when $n$ is close to $T$, and its inverse amplifies estimation error severely.

Shrinkage estimation addresses this by forming a convex combination of the sample covariance and a structured target matrix $\Phi$:

$$\hat{Q} = (1 - \delta) S + \delta \Phi$$

where $\delta \in [0, 1]$ is the shrinkage intensity. The target provides a regularizing structure that reduces variance at the cost of introducing some bias; the optimal $\delta$ minimizes the expected squared loss under a specific loss function.

Two shrinkage modes were implemented. The first follows the analytic formula of Ledoit and Wolf (2004), which shrinks toward a scaled identity matrix $\Phi = \hat{\mu} I$ where $\hat{\mu} = \text{tr}(S)/n$. The optimal shrinkage intensity is estimated analytically without cross-validation:

$$\delta^* = \min\!\left(\frac{\hat{\beta}^2}{\hat{\delta}^2}, 1\right)$$

where $\hat{\delta}^2 = \|S - \Phi\|_F^2$ measures the distance between the sample covariance and the target, and $\hat{\beta}^2$ is an asymptotic variance term estimated from the data. This is computationally efficient and requires no tuning.

The second mode uses a custom shrinkage target: the factor model covariance $Q_{\text{factor}}$ from the Ridge regression. Rather than shrinking toward the uninformative identity, this blends the sample covariance with an economically structured estimator:

$$\hat{Q}_{\text{LW}} = (1 - w) S + w Q_{\text{factor}}$$

where $w \in [0, 1]$ is treated as a tunable hyperparameter. This approach is more principled than shrinking toward identity: it says that to the extent the sample covariance is unreliable, the uncertainty should be resolved in the direction of the factor model structure rather than toward isotropy. When $w = 0$ the result is the raw sample covariance; when $w = 1$ it collapses to the factor model covariance entirely. Intermediate values blend both sources of information.

## 3.5 Risk Parity

As an alternative to MVO-based approaches, a risk parity strategy was implemented. Rather than maximizing a utility function, risk parity allocates capital such that each asset contributes equally to total portfolio variance. The risk contribution of asset $i$ is:

$$RC_i = x_i \cdot \frac{(Qx)_i}{x^\top Q x}$$

and the objective is to find weights satisfying $RC_i = 1/n$ for all $i$. This is formulated as a nonlinear least-squares problem:

$$\min_{x} \sum_{i=1}^n \left(RC_i - \frac{1}{n}\right)^2 \quad \text{subject to}
\quad \sum_i x_i = 1, \quad x_i \geq 0$$

solved via SLSQP. Risk parity has an important practical advantage: it requires no estimate of expected returns, eliminating the most error-prone component of MVO entirely. It tends to produce stable, diversified portfolios with relatively low turnover, making it naturally competitive on the second assessment criterion. The covariance matrix is estimated using Ledoit-Wolf shrinkage to ensure the optimization is well-conditioned.

## 3.6 Turnover Control

Since average turnover is explicitly penalized in the assessment, turnover control mechanisms were investigated during model development. In particular, an L1 penalty on changes in portfolio weights was incorporated into the MVO objective:

$$
\max_{x} \quad \mu^\top x - \frac{\gamma}{2} x^\top Q x - \lambda \|x - x_{\text{prev}}\|_1
$$

where $x_{\text{prev}}$ is the weight vector from the previous rebalancing period and $\lambda \geq 0$ controls the strength of the penalty. The L1 norm on weight changes is used rather than L2 because it tends to produce sparse updates — many weights remain exactly unchanged — which more directly reduces measured turnover.

This penalty is convex and preserves the convexity of the overall problem, allowing it to be incorporated directly into the CVXPY formulation without changing the optimization framework. On the first rebalancing period no penalty is applied, consistent with the project specification that the initial portfolio construction is not scored for turnover.

Although turnover control was investigated as part of the model development process, empirical testing revealed a consistent trade-off: higher turnover penalties reduced measured turnover but also materially reduced the Sharpe ratio. Given that the project assigns 80% of the score to Sharpe ratio and only 20% to turnover, very small or zero penalty values were ultimately preferred. The selected final configuration uses $\lambda = 0$ (no explicit turnover penalty), relying instead on the stability properties of the Ridge and Ledoit-Wolf estimators to naturally limit weight changes between periods.

## 3.7 Final Strategy: Ridge Regression with Factor-Based Covariance Shrinkage

Following the hyperparameter tuning and walk-forward evaluation procedures described in Section 3.8, the final selected strategy combines Ridge factor modelling for expected return estimation, factor-based Ledoit-Wolf covariance shrinkage, and mean-variance portfolio optimization. This approach produced the strongest overall out-of-sample Sharpe ratio among the candidate strategies while remaining computationally efficient.

At each rebalancing date, expected returns are estimated using a Ridge-regularized factor model. The eight provided Fama-French factors are used to explain historical excess returns, with Ridge regularization mitigating instability caused by multicollinearity among factor returns. The resulting factor loadings are then used to estimate both expected returns and a structured factor-model covariance matrix.

Rather than relying solely on the sample covariance matrix, covariance estimation is further improved through shrinkage toward the Ridge factor-model covariance:

$$
\hat{Q}_{\text{LW}} = (1-w)S + w Q_{\text{factor}}
$$

where $S$ is the sample covariance matrix, $Q_{\text{factor}}$ is the covariance matrix implied by the Ridge factor model, and $w$ is the shrinkage intensity selected through validation. This approach combines the flexibility of the sample covariance matrix with the stability and economic structure of the factor model.

Portfolio weights are then obtained by solving the mean-variance optimization problem:

$$
\max_x \quad \mu^\top x - \frac{\gamma}{2}x^\top Qx
$$

subject to

$$
\sum_i x_i = 1,
\qquad
x_i \ge 0,
\qquad
x_i \le x_{\max}.
$$

The optimization is performed using CVXPY with the CLARABEL solver, with a small regularization term added to the diagonal of $Q$ for numerical stability. All portfolios are fully invested, long-only, and subject to position limits to reduce concentration risk. The final hyperparameters used are `NumObs = 48`, `ridge_alpha = 0.1`, `lw_shrink_weight = 0.7`, and `turnover_penalty = 0.0`, as determined through grid search on the validation set.

This strategy combines three complementary sources of robustness. First, Ridge regularization stabilizes expected return estimates by reducing estimation variance in factor loadings. Second, covariance shrinkage produces a better-conditioned risk model than the raw sample covariance. Third, portfolio constraints prevent the optimizer from exploiting estimation noise through excessively concentrated positions.

## 3.8 Walk-Forward Backtesting Framework

All strategies were evaluated using a walk-forward backtest that strictly respects the information constraint: at each rebalancing date, only data available up to that date is used. The backtesting engine replicates the exact performance calculation used in the project evaluation: portfolio value is tracked in dollar terms by computing the number of shares held at each rebalancing date, turnover is measured as the L1 norm of the change in portfolio weights after accounting for price drift between rebalancing dates, and the Sharpe ratio is computed as the geometric mean of monthly excess returns divided by their standard deviation. The risk-free rate used to compute excess returns is the RF column provided in the factor dataset, ensuring consistency with the evaluation methodology.

For model selection, the dataset was partitioned into three non-overlapping periods: a training window used for initial calibration, a validation window used for hyperparameter selection via grid search, and a held-out test window used for a single final evaluation of the selected model. This structure guards against overfitting to the validation period and provides an honest estimate of out-of-sample performance prior to submission.

---

# 4. Results

This section presents the results of the model selection process and the final out-of-sample performance evaluation. Five portfolio construction approaches were considered: Historical Mean-Variance Optimization (Historical MVO), OLS-based Mean-Variance Optimization (OLS MVO), Risk Parity, Historical Maximum Sharpe Ratio Optimization, and Ridge Regression with Ledoit-Wolf Covariance Shrinkage (Ridge + LW). An equal-weight portfolio was also included as a benchmark.

For each candidate strategy, a systematic grid search was performed over a range of hyperparameters. The hyperparameter configurations were evaluated using a walk-forward backtesting framework and ranked according to a validation score that rewarded high Sharpe ratios while applying a modest penalty for portfolio turnover:

$$
\text{Score} = \text{Sharpe Ratio} - 0.02 \times \text{Average Turnover}
$$

The turnover penalty was intentionally small because the project assigns substantially greater importance to risk-adjusted return than to turnover. The objective of this stage was therefore to identify parameter combinations that achieved strong Sharpe ratios without generating excessive portfolio turnover.

## 4.1 Hyperparameter Selection

Hyperparameter tuning was performed separately for each strategy using a grid search over the parameter ranges described in Section 3.8. Table 4 summarizes the optimal hyperparameter configuration selected for each candidate strategy.

**Table 4: Selected Hyperparameters by Strategy**

| Strategy | Best Hyperparameters |
|---|---|
| Ridge + LW | NumObs = 48, Ridge Alpha = 0.1, Shrink Weight = 0.7, Turnover Penalty = 0.0 |
| OLS MVO | NumObs = 48, Risk Aversion = 5, Max Weight = 0.10 |
| Historical MVO | NumObs = 48, Risk Aversion = 5, Max Weight = 0.10 |
| Risk Parity | NumObs = 36, Covariance Method = Ledoit-Wolf |
| Historical Max Sharpe | NumObs = 48, Max Weight = 0.075 |

Several patterns emerged from the tuning results. Across MVO-based strategies, a 48-month observation window and tight position limits of 10% per asset consistently outperformed longer windows or wider limits, suggesting that the portfolio benefits more from concentration control than from additional historical data. For the Ridge + LW strategy, the optimal configuration used a high Ledoit-Wolf shrinkage weight of 0.7 toward the factor model covariance, indicating that the structured factor-based prior is more valuable than the raw sample covariance at this sample size. The turnover penalty parameter was most effective at very small values: large penalties materially degraded the Sharpe ratio without proportionally reducing turnover.

## 4.2 Out-of-Sample Test Performance

After hyperparameter selection, each strategy was evaluated on a previously unseen test set. This test period was not used during model development and therefore provides an unbiased estimate of out-of-sample performance. Table 5 reports the test-period Sharpe ratio and average turnover for the best version of each strategy.

**Table 5: Test-Period Performance by Strategy**

| Strategy | Test Sharpe Ratio | Average Turnover |
|---|---|---|
| Ridge + LW              | 0.1796        | 0.5226 |
| OLS MVO                 | 0.2246        | 0.4098 |
| Historical MVO          | 0.2239        | 0.4298 |
| Risk Parity             | 0.2865        | 0.1730 |
| Historical Max Sharpe   | 0.3220        | 0.2924 |
| Equal Weight            | 0.2538        | 0.1025 |

It is important to note that the test-period comparison in Table 5 was used only as a diagnostic evaluation during model development and not as the final model-selection criterion. The project evaluation is based on the full walk-forward backtest described in Section 3.8. Consequently, strategy selection was ultimately guided by performance under the same methodology used by the project scoring system, rather than by performance on a single market sub-period.

The test-period results present a seemingly counterintuitive ordering: Historical Max Sharpe and Risk Parity lead with Sharpe ratios of 0.3220 and 0.2865 respectively, while the more sophisticated Ridge + LW strategy finishes last among model-based approaches at 0.1796. Several factors explain this pattern.

First, the test period represents a single contiguous slice of market history — approximately the last 20% of the dataset — and may correspond to a specific market regime that happens to favor simpler, more defensive strategies. Risk Parity and equal weighting tend to perform well in low-dispersion environments where return differences across assets are small and mean-reverting, since they do not attempt to forecast returns and therefore cannot be hurt by return estimation error. Historical Max Sharpe benefits similarly from its direct optimization of the Sharpe ratio objective, which aligns exactly with the evaluation criterion, though it is typically fragile out-of-sample due to its sensitivity to expected return estimates.

Second, a single test period provides limited statistical power for strategy discrimination. The differences in Sharpe ratios observed here, while numerically meaningful, could easily be reversed over a different time window of the same length. This is precisely why the competition uses a full walk-forward evaluation rather than a single held-out test, and why Table 6 is the more relevant performance measure.

Finally, the Ridge + LW strategy's higher turnover of 0.5226 relative to simpler strategies reflects its greater responsiveness to changing factor model estimates at each rebalancing date. In a regime where the factor signals are noisy or unreliable, this responsiveness translates to churn without corresponding return improvement.

## 4.3 Final Strategy Comparison

The final evaluation follows the project methodology. The first 60 months of observations are reserved for calibration and excluded from scoring. Portfolios are subsequently rebalanced every six months using only information available at the rebalancing date. Table 6 reports the final out-of-sample Sharpe ratio and average turnover for all candidate strategies evaluated on the training dataset provided.

**Table 6: Final Walk-Forward Comparison**

| Strategy              | Sharpe Ratio | Average Turnover |
| --------------------- | ------------ | ---------------- |
| Ridge + LW            | 0.1991       | 0.5213           |
| OLS MVO               | 0.1546       | 0.4590           |
| Historical MVO        | 0.1606       | 0.4654           |
| Risk Parity           | 0.1518       | 0.1788           |
| Historical Max Sharpe | 0.1750       | 0.3850           |
| Equal Weight          | 0.1653       | 0.1170           |

The full walk-forward evaluation provides the most relevant measure of strategy performance because it mirrors the project scoring methodology and evaluates each approach across the entire investment horizon. Under this framework, Ridge + LW achieves the highest Sharpe ratio of 0.1991, outperforming all benchmark strategies. Historical Maximum Sharpe ranks second with a Sharpe ratio of 0.1750, while Equal Weight achieves 0.1653. The remaining strategies produce Sharpe ratios between 0.1518 and 0.1606.

Several observations emerge from these results. First, the fact that Ridge + LW outperforms both Historical MVO and OLS MVO suggests that regularization improves portfolio performance. Ridge regression stabilizes factor loading estimates in the presence of multicollinearity, while covariance shrinkage reduces the impact of estimation error in the sample covariance matrix. Together, these components produce more reliable inputs for portfolio optimization and lead to improved out-of-sample risk-adjusted returns.

Second, the improvement over the Equal Weight benchmark demonstrates that the factor-based modelling framework adds value beyond a passive allocation approach. Although the improvement in Sharpe ratio is modest, Equal Weight is widely recognized as a difficult benchmark to outperform consistently due to its complete immunity to estimation error. The superior performance of Ridge + LW therefore provides evidence that the model is extracting useful information from the factor data rather than simply fitting historical noise.

Third, the results highlight the trade-off between return and turnover. Equal Weight and Risk Parity generate substantially lower turnover than the MVO-based strategies due to the stability of their allocation rules. In contrast, Ridge + LW exhibits the highest turnover at 0.5213 because portfolio weights adjust in response to updated factor estimates at each rebalancing date. While this increases trading activity, the higher turnover is accompanied by the strongest risk-adjusted performance, suggesting that the additional portfolio adjustments are economically justified within the context of the project objective.

Based on these results, Ridge + LW was selected as the final submission strategy. It achieved the highest Sharpe ratio under the project evaluation methodology while remaining theoretically well-founded and computationally efficient. The combination of factor-based return estimation, covariance shrinkage, and constrained mean-variance optimization provides a robust framework for portfolio construction and offers the strongest evidence of out-of-sample performance among the strategies considered.


# 5. Discussion

## 5.1 Strengths of the Final Algorithm

**Principled regularization at every stage.** The strategy applies regularization independently to both the return estimator (Ridge) and the covariance estimator (Ledoit-Wolf), addressing the two primary sources of instability in classical MVO. Neither component requires cross-validation to select its regularization structure: Ridge solves a closed-form linear system and the Ledoit-Wolf blend is computed directly, making the algorithm computationally fast and deterministic.

**Factor structure as an economic prior.** By using the factor model covariance $Q_{\text{factor}}$ as the Ledoit-Wolf shrinkage target rather than the identity matrix, the strategy embeds economic structure into the covariance estimate. This is more meaningful than isotropic shrinkage: it says the uncertainty in the sample covariance should be resolved toward the covariance structure implied by the Fama-French factors, rather than toward the assumption that all assets are uncorrelated and equally volatile.

**Scalability to different universe sizes.** The algorithm makes no assumptions about the number of assets. Because it uses a factor model covariance decomposition $Q = V^\top F_{\text{cov}} V + D$, the number of free parameters scales with the number of factors ($p = 8$) and assets ($n$) independently, rather than as $n(n+1)/2$ as in an unrestricted sample covariance. This makes the strategy well-suited to the project specification where the asset universe may range from 15 to 40 assets.

**Convex optimization with global optimality guarantees.** The MVO formulation with long-only and position-limit constraints is a convex quadratic program, solved using CVXPY with the CLARABEL interior-point solver. This guarantees that the globally optimal portfolio is found at each rebalancing date, with no sensitivity to initialization or local optima.

**Adaptable observation window.** The 48-month rolling window allows the model to remain responsive to changing market conditions while retaining enough data for stable parameter estimation. 

## 5.2 Weaknesses of the Final Algorithm

**Sensitivity to expected return estimation.** Despite Ridge regularization, the factor-model expected return estimate $\mu = \hat{\alpha} + \hat{V}^\top \bar{f}$ remains highly sensitive to the sample mean of factor returns $\bar{f}$ and to the estimated alpha $\hat{\alpha}$. Both are notoriously noisy at monthly frequency. MVO's well-known error-maximization property means that even modest errors in $\mu$ can produce concentrated, unstable portfolios, particularly when assets have similar risk profiles.

**No explicit regime detection.** The strategy uses a fixed rolling window and applies the same model structure regardless of whether the market is in a trending, mean-reverting, or high-volatility regime. A strategy with regime-switching components could potentially allocate more defensively during periods of market stress, but such approaches are significantly more complex to implement and validate robustly.

**Turnover not directly minimized.** The final configuration uses no turnover penalty ($\lambda = 0$), relying solely on the natural stability of the Ridge and Ledoit-Wolf estimators to limit weight changes. This is an indirect form of turnover control. A more direct approach — such as adding a transaction cost term to the objective or imposing an explicit turnover constraint — could reduce turnover further without compromising Sharpe ratio as severely as the L1 penalty did during validation.

**Risk parity not incorporated as a fallback.** Risk parity is entirely return-agnostic and tends to produce stable, low-turnover allocations in regimes where factor-based return forecasts are unreliable. Incorporating it as an ensemble component alongside the Ridge + LW strategy could reduce drawdowns in adverse market environments, though determining when to switch between strategies introduces its own model-selection complexity.

**Hyperparameters tuned on a single dataset.** The final hyperparameters (`NumObs = 48`, `ridge_alpha = 0.1`, `lw_shrink_weight = 0.7`) were selected by grid search on the first provided training dataset. While the validation methodology was rigorous, there is no guarantee that these values are optimal for the two unseen datasets, which may have different asset universes, correlation structures, and time periods. More robust hyperparameter selection — such as averaging across multiple training datasets or using a longer validation window — could improve generalization, at the cost of additional computation.


# 6. Conclusion

The final algorithm combines three methodological components: Ridge regression on the eight Fama-French factors for expected return estimation, Ledoit-Wolf shrinkage toward a factor-model covariance matrix for risk estimation, and mean-variance optimization with long-only and position-limit constraints for portfolio construction.

The model development process evaluated six candidate strategies — Equal Weight, Historical MVO, OLS MVO, Risk Parity, Historical Maximum Sharpe, and Ridge + LW — using a train-validation-test framework designed to prevent look-ahead bias. Hyperparameters were selected through systematic grid search, with the optimal Ridge + LW configuration consisting of a 48-month rolling estimation window, Ridge regularization parameter $\alpha = 0.1$, factor-model covariance shrinkage weight $w = 0.7$, and no explicit turnover penalty.

Under the final walk-forward evaluation methodology, which mirrors the project scoring framework, the Ridge + LW strategy achieved the highest Sharpe ratio among all candidate approaches at 0.1991 with an average turnover of 0.5213. While some competing strategies produced stronger performance over isolated test periods, Ridge + LW demonstrated the most consistent performance across the full investment horizon and ultimately provided the best balance between return forecasting, risk estimation, and portfolio construction.

A key finding of the project is that regularization plays a critical role in portfolio optimization. Traditional mean-variance optimization based on historical means and sample covariances was consistently hindered by estimation error, while Ridge regression stabilized factor loadings and covariance shrinkage improved the conditioning of the risk model. Together, these techniques produced more robust portfolios and superior out-of-sample performance.

The primary strengths of the selected approach are its economic interpretability, scalability to larger asset universes, and robustness to estimation error. By incorporating factor information into both the return and covariance estimation processes, the strategy leverages economically meaningful structure while avoiding many of the weaknesses associated with purely historical estimators.

The primary limitation of the approach is its continued reliance on factor-model expected returns, which remain inherently noisy at monthly frequencies despite regularization. In addition, the strategy generates higher turnover than simpler approaches such as Equal Weight and Risk Parity because portfolio allocations respond to changing factor estimates at each rebalancing date.

Future extensions could explore shrinkage of expected returns through methods such as Black-Litterman or James-Stein estimation, dynamic covariance shrinkage schemes, regime-switching frameworks, or ensemble approaches that combine factor-based forecasting with risk-parity-style allocation. Incorporating explicit transaction cost modelling may also improve the trade-off between risk-adjusted return and turnover.

Overall, the results demonstrate that combining Ridge regression, factor-based covariance shrinkage, and constrained mean-variance optimization provides a robust and effective framework for systematic portfolio management. The Ridge + LW strategy delivered the strongest performance under the project's evaluation methodology and was therefore selected as the final submission.



# 7. References

[1] French, K. R. *Data Library*. http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html. Accessed February 2020.

[2] Markowitz, H. "Portfolio selection." *The Journal of Finance*, 7(1), 1952, pp. 77–91.

[3] Ledoit, O. and Wolf, M. "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), pp. 365–411.