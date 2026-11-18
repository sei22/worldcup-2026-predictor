# 🏆 World Cup 2026 — Poisson + Monte-Carlo Predictor

Predicting the 2026 FIFA World Cup by fitting a **Poisson regression** on international match
results, then replaying the whole tournament **1,000 times** with Monte-Carlo simulation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![statsmodels](https://img.shields.io/badge/statsmodels-Poisson%20GLM-orange?style=for-the-badge)
![Method](https://img.shields.io/badge/Method-Monte%20Carlo-5a2d82?style=for-the-badge)

📓 **[Open the notebook](worldcup_2026_model.ipynb)**

## The idea

Most football prediction models stop at "who wins". A Poisson regression goes further: it estimates
the **expected number of goals** for each side, which gives a full distribution over exact scores —
and from there, win/draw/loss probabilities, goal difference, group standings, and knockout odds.

The model learns from every international match since 2021 (5,821 of them) using five families of
features:

| Feature | Idea | Fitted coefficient |
|---|---|---|
| `diff_elo` | Elo rating gap between the two teams | **+0.0018** *** |
| `est_domicile` | the team plays in its own country | **+0.2024** *** |
| `diff_momentum_pts` | points won over the last 10 matches | −0.0108 *** |
| `diff_momentum_bp` | goals scored over the last 10 matches | +0.0076 *** |
| `diff_momentum_bc` | goals conceded over the last 10 matches | −0.0053 *** |
| `penalite_alt` | oxygen debt: stadium altitude minus the team's usual altitude | −0.0817 ** |
| `choc_thermique` | gap between stadium temperature and the team's usual climate | −0.0016 (n.s.) |

*** p < 0.001, ** p < 0.01, n.s. = not significant.

Two findings worth noting. **Home advantage is worth about 22% more goals** (e^0.2024), which matches
the football literature. **Altitude is statistically significant and penalises the visiting side** —
playing 1,000 m above your usual altitude costs roughly 8% of your expected goals. The thermal shock
coefficient, on the other hand, is indistinguishable from zero: in this dataset, heat does not measurably
change scorelines.

## Design decisions

**Each match produces two observations**, one per team, with the symmetric features sign-flipped
(`diff_elo` becomes `-diff_elo`). Altitude and temperature are the exception — they are a burden on
whoever is away from home, not a zero-sum quantity, so they are never flipped.

**Momentum uses `.shift(1)`.** Rolling statistics are computed over the 10 matches *preceding* each
fixture. Without that shift, the current result would leak into its own predictors and the model's
measured accuracy would be fiction.

**The simulation bypasses `model.predict()`.** Running 1,000 full tournaments means hundreds of
thousands of match evaluations; calling statsmodels each time is far too slow. The coefficients are
extracted once and λ is computed directly as `exp(β·x)`, which turns hours into seconds.

**The validation split is chronological, not random.** A random split would let the model train on
matches played after the ones it is asked to predict.

## Results

Title odds over 1,000 simulated tournaments (top 10):

| Team | Reach R16 | Quarters | Semis | Final | **Title** |
|---|---|---|---|---|---|
| 🇪🇸 Spain | 100% | 70% | 53% | 40% | **28.8%** |
| 🇦🇷 Argentina | 97% | 55% | 38% | 29% | **17.1%** |
| 🇫🇷 France | 96% | 39% | 23% | 16% | **8.5%** |
| 🇧🇷 Brazil | 96% | 50% | 34% | 16% | **7.1%** |
| 🇵🇹 Portugal | 94% | 36% | 17% | 10% | **5.1%** |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 97% | 40% | 16% | 10% | **4.7%** |
| 🇨🇴 Colombia | 92% | 34% | 16% | 9% | **3.4%** |
| 🇳🇱 Netherlands | 93% | 36% | 20% | 8% | **3.3%** |
| 🇩🇪 Germany | 94% | 38% | 20% | 7% | **2.9%** |
| 🇳🇴 Norway | 87% | 28% | 13% | 6% | **2.8%** |

## Validation

**Match-level accuracy: 61.93%** on 1,169 held-out matches (1-X-2 classification, chronological split).

The confusion matrix shows the classic weakness of independent Poisson models: draws are badly
predicted (36 correct out of 270). A ×1.35 correction is applied to the draw probability to partially
offset the bias, but the model still resolves most matches into a winner.

**Out-of-sample back-test — Qatar 2022.** The model is retrained on 2018 → 19 November 2022 only, the
eve of the opening match, then asked to simulate a tournament it has never seen. Section 13 scores its
predicted group qualifiers against the sixteen teams that actually made it through.

## Data

Everything is downloaded at runtime — **no dataset is stored in this repository**:

* [eloratings.net](https://www.eloratings.net) — match results and pre-match Elo ratings from 2018 onwards, plus the live world ranking
* [Our World in Data](https://ourworldindata.org/grapher/average-annual-surface-temperature) — average annual surface temperature per country
* Country altitudes are embedded in the notebook as a parsed lookup table

## Running it

```bash
pip install -r requirements.txt
jupyter notebook worldcup_2026_model.ipynb
```

Then run the cells in order. The notebook also opens directly in Google Colab, which is where it was
developed. A full run takes a few minutes, dominated by the two Monte-Carlo sections (1,000 tournaments
each); lower `NB_SIMULATIONS` in section 1 for a quicker pass.

## Limitations

* **Poisson independence.** Home and away goals are modelled as independent draws, which is why draws
  are under-predicted. A bivariate Poisson or a Dixon-Coles correction would address this properly.
* **No squad information.** Injuries, suspensions, and selections are invisible to the model; Elo and
  recent form are the only proxies for team strength.
* **Approximate geography.** Stadium temperatures are summer averages, not match-day forecasts, and each
  group is pinned to a single base-camp city rather than its real fixture list.
* **Snapshot in time.** The model was built on data up to early June 2026, before the tournament. The
  probabilities above are a genuine forward-looking forecast as of that date, not a retrospective fit.

## Author

**Sei Bayle** — personal project, built during an exchange semester at Universitas Gadjah Mada.
