# DFSignal Business Glossary

This glossary keeps the exact technical meaning while giving business users a practical interpretation.

| Term | Technical definition | Plain-English explanation | Simple example | Why it matters |
|---|---|---|---|---|
| Forecast | An estimate of future demand from a model. | A best estimate of what demand may be. | Next week's estimate is 1,000 units. | Supports planning, but is not a promise. |
| Baseline | A reference method used for comparison. | The simpler answer we must beat before adding complexity. | Seasonal Naive repeats a comparable seasonal pattern. | Prevents signals from being called useful without evidence. |
| Seasonality | Repeating changes linked to calendar or time of year. | A pattern that tends to repeat. | Demand is usually higher before holidays. | Helps distinguish recurring timing from new change. |
| Trend | Sustained movement up or down over time. | The general direction demand is moving. | Demand rises gradually for six months. | Influences longer-range planning. |
| Lag | A past value used as an input. | What happened one or more periods ago. | Last week's demand is a one-week lag. | Recent history often contains useful information. |
| Rolling Average | Average over a moving recent window. | A smoother view of recent demand. | The four-week average reduces week-to-week noise. | Makes the underlying pattern easier to use. |
| Backtest | Historical simulation where past data is used to predict later observed data. | A rehearsal using history. | Train through June and test July. | Shows how a method behaved before today. |
| WAPE | Weighted Absolute Percentage Error: total absolute error divided by total actual demand. | How far the forecast was from actual demand overall. | WAPE = 8% means it was off by about 8% overall. | Lower is generally better. |
| MAE | Mean Absolute Error: average absolute difference between actual and predicted values. | Average miss in demand units. | MAE = 12 means the average miss was 12 units. | Keeps results in business units. |
| RMSE | Root Mean Squared Error: square root of average squared errors. | An error measure that penalizes large misses more. | One very large miss affects RMSE strongly. | Highlights costly surprises. |
| MASE | Mean Absolute Scaled Error relative to a reference error. | Error compared with a simple reference pattern. | MASE = 1 means similar error to the reference. | Enables comparisons across series and scales. |
| Bias | Signed average forecast error; in DFSignal, actual minus forecast. | Whether the forecast tends to be too high or too low. | Negative bias means predictions tended to be too high. | Reveals a consistent planning direction. |
| Forecast Value Add (FVA) | Baseline WAPE minus model WAPE. | Whether an added method or signal improved accuracy. | Positive FVA means lower error than the baseline. | Tests whether complexity earns its place. |
| Prediction Interval | Lower and upper forecast bounds calibrated from residuals. | A range showing uncertainty around the estimate. | Demand may be between 900 and 1,100. | Prevents false precision. |
| Feature | An input variable supplied to a model. | A piece of information used to make a forecast. | Recent demand or a launch count. | Defines what evidence the model can use. |
| External Signal | Input sourced outside the target demand series. | Market information that may provide context. | GPU launches or Steam adoption. | May help, but must prove value on relevant data. |
| GPU Adoption | Share of Steam survey users reporting a GPU generation. | How widely a GPU generation appears among surveyed gamers. | RTX 40 share rises from 5% to 8%. | Can indicate upgrade cycles, not direct sales. |
| Model | A repeatable calculation method that turns inputs into predictions. | The forecasting approach. | ETS or LightGBM. | Different approaches can behave differently. |
| LightGBM | Gradient-boosted decision-tree implementation used here for forecasting. | A flexible model that combines history and optional signals. | It can use lags, calendar values, and GPU inputs. | Flexible models can also overfit or be unstable. |
| ETS | Exponential smoothing model using level, trend, and seasonality. | A history-based forecasting method. | It extends recent patterns into the future. | Strong reference for recurring demand patterns. |
| Seasonal Naive | Forecast that repeats a prior seasonal value or pattern. | Assume demand follows a similar pattern to the previous seasonal period. | Use the comparable week last year. | Provides a simple, understandable baseline. |
| SHAP | Model-attributed contribution measure for individual features. | Which inputs the model relied on most, according to the model. | History accounts for most attributed influence. | Explains model behavior; it does not prove cause. |
| Observed Only | Future external values are limited to values observed by the forecast origin. | Do not invent future market data. | No future Steam rows are fabricated. | Protects the test from looking unrealistically informed. |
| Known Future | Input values legitimately available before the prediction date. | Information planners really could know in advance. | A published holiday calendar. | Avoids giving the model information it would not have. |
| Forecast Horizon | Number of periods between forecast origin and target period. | How far ahead the estimate reaches. | A 5–8 week horizon. | Accuracy and useful signals can change with distance. |
| Data Leakage | Accidental use of information unavailable at forecast time. | Letting the answer leak into the test. | Using a future month's adoption value. | Makes performance look better than it would be in practice. |
| Fold | One train-and-test slice in a rolling backtest. | One historical rehearsal. | Five folds means five test periods for that horizon. | Consistency across folds matters more than one lucky result. |
| Mapping Coverage | Share of source rows matched to a known category or entity. | How much source information was classified. | 50% of rows mapped to a GPU generation. | Unmapped data limits interpretation. |
| Share-Weighted Coverage | Source coverage weighted by each observation's reported share. | How much of the important observed volume was classified. | 66% share-weighted coverage can exceed 50% row coverage. | More useful than row coverage when rows have unequal importance. |
| Provenance | Record of source, transformation, timing, and ownership. | Where information came from and what happened to it. | A source URL and load timestamp. | Makes results traceable and reviewable. |
| Synthetic Data | Artificially generated data designed to resemble a process. | Test data, not real customer behavior. | Seeded weekly demand from 2016–2025. | Useful for engineering checks but not business proof. |

## Standard confidence wording

- **Strong evidence:** The result was consistent across most historical test periods.
- **Mixed evidence:** The signal helped in some periods but hurt in others.
- **Limited evidence:** The result is based on limited data or unstable performance.
- **Not yet tested:** The pipeline supports this capability, but it has not yet been validated.

## Gate and data-quality wording

- **GO:** The system is ready for the next validation stage.
- **CONDITIONAL_GO:** The system is ready to continue testing, but important questions remain.
- **STOP_AND_FIX:** Results should not be used until the identified issue is corrected.
- **PASS:** No material data issue was detected.
- **REVIEW:** The data can be used for testing, but some quality issues should be reviewed.
- **FAIL:** The issue is significant enough that results should not be trusted yet.
- **MISSING_OPTIONAL:** This optional dataset is not currently available, but the rest of the pipeline can still run.
