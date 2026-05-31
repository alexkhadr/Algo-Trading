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

The objective of this project is to design and implement an automated asset management system — an algorithmic trading strategy capable of constructing and rebalancing a portfolio of equities and equity-based ETFs on a semi-annual basis. The algorithm is assessed primarily on two out-of-sample financial metrics: the ex-post Sharpe ratio, which measures risk-adjusted return over the full investment horizon, and the average portfolio turnover rate, which measures the magnitude of weight changes at each rebalancing date. These dual objectives create an inherent tension: strategies that aggressively respond to new information tend to generate higher turnover, while strategies that minimize trading may sacrifice return. A well-designed algorithm must therefore balance both.

The dataset consists of monthly adjusted closing prices for a universe of equities and ETFs, paired with returns on eight factors — market, size, value, short-term reversal, profitability, investment, momentum, and long-term reversal — along with the contemporaneous risk-free rate. The first 60 months of each dataset are reserved exclusively for initial calibration and are not included in the out-of-sample performance evaluation. Portfolios are rebalanced every six months using a walk-forward methodology, meaning the algorithm is only permitted to use data available up to each rebalancing date, with no lookahead.

Given these constraints, the model development process focused on three interconnected problems. The first is return estimation: how to form reliable forward-looking expected returns from noisy historical data and factor exposures. The second is covariance estimation: how to construct a well-conditioned covariance matrix suitable for optimization, particularly given the limited number of observations relative to the number of assets. The third is portfolio construction: how to translate estimates of return and risk into a portfolio that maximizes risk-adjusted performance while keeping turnover low enough to remain competitive on the second criterion.

To address these problems, several modelling approaches were developed, implemented, and evaluated through a systematic train-validate-test framework before a final strategy was selected. The candidate strategies ranged from simple benchmarks — equal weighting and historical mean-variance optimization — to more sophisticated approaches combining regularized factor models with shrinkage-based covariance estimation and explicit turnover control.

# 2. Data

## 2.1 Asset Price Data

The investment universe consists of 20 U.S. stocks whose tickers are listed in Table 1. The dataset provides monthly adjusted closing prices for each stock from December 2001 to December 2016. These prices are used to compute monthly asset returns.

**Table 1: Investment Universe**

| F | CAT | DIS | MCD | KO | PEP | WMT | C | WFC | JPM |
|---|-----|-----|-----|----|-----|-----|---|-----|-----|
| AAPL | IBM | PFE | JNJ | XOM | MRO | ED | T | VZ | NEM |

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

The presence of factor correlations also has implications for return estimation. In OLS, correlated regressors inflate the variance of individual coefficient estimates, making individual factor loadings unreliable even when the joint fit is good. This is the primary motivation for replacing OLS with Ridge regression in the final strategy, as discussed in Section 2.3.

## 2.3 Calibration and Investment Windows

The full training dataset spans December 2001 to December 2016. Per the project specifications, the first 60 months — January 2002 through December 2006 — are reserved exclusively for the initial calibration period and are not included in the out-of-sample performance evaluation. Out-of-sample performance is therefore measured from January 2007 onward.

The portfolio is rebalanced every six months. At each rebalancing date, the strategy is recalibrated using all data available up to that point, consistent with an expanding window approach. Alternatively, the rolling window length $T_0$ is a tunable hyperparameter: the algorithm uses only the most recent $T_0$ months for estimation at each rebalancing date, allowing older and potentially less relevant data to be discarded. The effect of $T_0$ on out-of-sample performance is evaluated during the grid search described in Section 3.

For the purposes of model selection, the out-of-sample period is further divided into three non-overlapping segments as summarized in Table 3. The training segment is used only for initial calibration runs; the validation segment is used for hyperparameter selection via grid search; and the test segment is evaluated exactly once using the best parameters identified on the validation set.

**Table 3: Data Partitioning for Model Selection**

| Segment | Period | Purpose |
|---------|--------|---------|
| Initial calibration | Jan 2002 – Dec 2006 | Reserved; not scored |
| Training | Jan 2007 – XXX | Strategy calibration during grid search |
| Validation | XXX - XXX | Hyperparameter selection |
| Test | XXX - XXX | Final evaluation |

This three-way split is necessary to avoid a subtle but consequential form of overfitting: if hyperparameters are selected by evaluating performance on the same data used to calibrate the model, the reported performance will be optimistic. By holding out the test segment entirely until after hyperparameters are finalized, the test Sharpe ratio and turnover provide an unbiased estimate of how the strategy is likely to perform on the two unseen datasets.


# 3. Methodology

## 3.1 Problem Formulation

At each rebalancing date $t$, the algorithm observes all available historical asset returns and factor returns up to and including period $t$, and must produce a portfolio weight vector $x \in \mathbb{R}^n$ satisfying:

$$\sum_{i=1}^n x_i = 1, \qquad x_i \geq 0, \qquad x_i \leq x_{\max}$$

The long-only constraint eliminates short positions, which is standard for an equity fund operating under typical institutional constraints. The per-asset cap $x_{\max} = 0.25$ prevents excessive concentration in any single position.

The core optimization problem is mean-variance optimization (MVO), originally formulated by Markowitz (1952). In its general form, the portfolio is selected to maximize a utility function that trades off expected return against portfolio variance:

$$\max_{x} \quad \mu^\top x - \frac{\gamma}{2} x^\top Q x$$

where $\mu \in \mathbb{R}^n$ is the vector of expected asset returns, $Q \in \mathbb{R}^{n \times n}$ is the covariance matrix of returns, and $\gamma > 0$ is a risk aversion parameter controlling the return-risk trade-off. This is a convex quadratic program (QP), solved efficiently using CVXPY with the CLARABEL solver.

A well-known limitation of MVO is its sensitivity to estimation error in $\mu$ and $Q$. Small perturbations in these inputs can produce large and unstable changes in the optimal weights. Much of the methodology described below is motivated by mitigating this instability — both through better estimation of $\mu$ and $Q$, and through explicit regularization of the optimization itself.

## 3.2 Benchmark Strategies

Three benchmark strategies were implemented to provide a performance baseline and isolate the contribution of each modelling component.

**Equal weighting** allocates $x_i = 1/n$ to each asset regardless of any data. While naive, equal weighting is known to be surprisingly competitive out-of-sample due to its complete immunity to estimation error, and it trivially achieves zero turnover after the initial allocation (since weights drift with prices and are reset symmetrically at each rebalance). It serves as a floor: any model-based strategy that underperforms equal weighting offers no value over a purely passive approach.

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

Even with a factor model, covariance matrix estimation remains a challenge. The sample covariance matrix $S$ is an unbiased estimator of $Q$, but it is known to be a poor estimator in finite samples: its eigenvalues are systematically dispersed relative to the true eigenvalues, it may be singular or nearly singular when $n$ is close to $T$, and its inverse — which appears in many portfolio optimization formulas — amplifies estimation error severely.

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

$$\min_{x} \sum_{i=1}^n \left(RC_i - \frac{1}{n}\right)^2 \quad \text{subject to} \quad \sum_i x_i = 1, \quad x_i \geq 0$$

solved via SLSQP. Risk parity has an important practical advantage: it requires no estimate of expected returns, eliminating the most error-prone component of MVO entirely. It tends to produce stable, diversified portfolios with relatively low turnover, making it naturally competitive on the second assessment criterion. The covariance matrix is estimated using Ledoit-Wolf shrinkage to ensure the optimization is well-conditioned.

## 3.6 Turnover Control

Since average turnover is explicitly penalized in the assessment, the final strategy incorporates a direct turnover control mechanism. At each rebalancing date, an L1 penalty on the change in weights is added to the MVO objective:

$$\max_{x} \quad \mu^\top x - \frac{\gamma}{2} x^\top Q x - \lambda \|x - x_{\text{prev}}\|_1$$

where $x_{\text{prev}}$ is the weight vector from the previous rebalancing period and $\lambda \geq 0$ controls the strength of the penalty. The L1 norm on weight changes is used rather than L2 because it tends to produce sparse updates — many weights remain exactly unchanged — which more directly reduces measured turnover. This penalty is convex and preserves the convexity of the overall problem, so it integrates cleanly into the CVXPY formulation without any change in solver or computational complexity. On the first rebalancing period no penalty is applied, consistent with the specification that the initial portfolio construction is not scored for turnover.

## 3.7 Final Strategy: XXX

description of final selected strategy

## 3.8 Walk-Forward Backtesting Framework

All strategies were evaluated using a walk-forward backtest that strictly respects the information constraint: at each rebalancing date, only data available up to that date is used. The backtesting engine replicates the exact performance calculation used in the competition: portfolio value is tracked in dollar terms by computing the number of shares held at each rebalancing date, turnover is measured as the L1 norm of the change in portfolio weights after accounting for price drift between rebalancing dates, and the Sharpe ratio is computed as the geometric mean of excess monthly returns divided by their standard deviation, annualized by $\sqrt{12}$. The risk-free rate used to compute excess returns is the RF column provided in the factor dataset, ensuring consistency with the evaluation methodology.

For model selection, the dataset was partitioned into three non-overlapping periods: a training window used for initial calibration, a validation window used for hyperparameter selection via grid search, and a held-out test window used for a single final evaluation of the selected model. This structure guards against overfitting to the validation period and provides an honest estimate of out-of-sample performance prior to submission.


# 4. Results

model selection process and testing

# 5. Discussion

outlining the strengths and weaknesses of your algorithm and why you chose it

# 6. Conclusion



# 7. References

[1] French, K. R. *Data Library*. http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html. Accessed February 2020.

[2] Markowitz, H. "Portfolio selection." *The Journal of Finance*, 7(1), 1952, pp. 77–91.

[3] Ledoit, O. and Wolf, M. "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), pp. 365–411.