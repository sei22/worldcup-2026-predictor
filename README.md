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

The model learns from every international match played between 2021 and the eve of the tournament
(5,764 of them) using five families of features:

| Feature | Idea | Fitted coefficient |
|---|---|---|
| `diff_elo` | Elo rating gap between the two teams | **+0.0017** *** |
| `est_domicile` | the team plays in its own country | **+0.2031** *** |
| `diff_momentum_pts` | points won over the last 10 matches | −0.0108 *** |
| `diff_momentum_bp` | goals scored over the last 10 matches | +0.0077 *** |
| `diff_momentum_bc` | goals conceded over the last 10 matches | −0.0054 *** |
| `penalite_alt` | oxygen debt: stadium altitude minus the team's usual altitude | −0.0790 ** |
| `choc_thermique` | gap between stadium temperature and the team's usual climate | −0.0014 (n.s.) |

*** p < 0.001, ** p < 0.01, n.s. = not significant.

Two findings worth noting. **Home advantage is worth about 23% more goals** (e^0.2031), which matches
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

**The dataset is frozen at 5 June 2026, the eve of the opening match, and the random seed is fixed.**
The sources are still queried live, so re-running the notebook today would otherwise feed it the World
Cup's own results — England's Elo alone gained 161 points during the tournament — and the forecast would
quietly become a retrodiction. The cutoff keeps it a forecast, and the seed makes every run return the
same figures.

## Results

Title odds over 1,000 simulated tournaments (top 10):

| Team | Reach R16 | Quarters | Semis | Final | **Title** |
|---|---|---|---|---|---|
| 🇪🇸 Spain | 100% | 69% | 51% | 40% | **28.0%** |
| 🇦🇷 Argentina | 99% | 57% | 38% | 29% | **17.8%** |
| 🇫🇷 France | 96% | 42% | 24% | 17% | **9.7%** |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 97% | 45% | 21% | 14% | **7.7%** |
| 🇳🇱 Netherlands | 94% | 46% | 28% | 12% | **5.9%** |
| 🇧🇷 Brazil | 97% | 49% | 29% | 12% | **5.3%** |
| 🇵🇹 Portugal | 92% | 33% | 15% | 8% | **3.2%** |
| 🇨🇴 Colombia | 93% | 31% | 14% | 7% | **3.1%** |
| 🇳🇴 Norway | 87% | 27% | 13% | 6% | **2.5%** |
| 🇪🇨 Ecuador | 96% | 35% | 21% | 7% | **2.4%** |

## Retrospective: how it actually went

The model was built in early June 2026, before a ball was kicked. The tournament has since been played:
**Spain beat Argentina 1-0 after extra time** in the final on 19 July at MetLife Stadium, for their
second world title.

| What happened | What the model had said |
|---|---|
| 🏆 **Champion: Spain** | ranked **1st**, at 28.0% |
| **Final: Spain vs Argentina** | the model's **exact top 2** by final probability (40% and 29%) |
| **Semi-finalists: Spain, Argentina, France, England** | ranked 1st, 2nd, 5th and 6th |
| **Quarter-finalists** (8 teams) | all eight inside the model's top 19 of 48; seven inside the top 15 |

**Brier score on the title: 0.577**, against 0.979 for a uniform 1-in-48 guess — a **41% improvement**
over knowing nothing. Section 14 of the notebook recomputes all of this directly from the simulation
it just ran, so the table above cannot drift away from the code.

A word on how to read this. A model that gives Spain 28.8% and then watches Spain win was not "right":
it said Spain would win roughly one time in three, and that happened. One tournament can never separate
a good model from a lucky one, which is why the Brier score — scoring the whole distribution rather than
the winner alone — matters more than the headline. The honest summary is that the model ranked the
eventual finalists first and second out of forty-eight teams, and that its probability mass sat where
the results landed.

## Validation

**Match-level accuracy: 61.92%** on 1,158 held-out matches (1-X-2 classification, chronological split).

The confusion matrix shows the classic weakness of independent Poisson models: draws are badly
predicted (35 correct out of 264). A ×1.35 correction is applied to the draw probability to partially
offset the bias, but the model still resolves most matches into a winner.

**Out-of-sample back-test — Qatar 2022.** The model is retrained on 2018 → 19 November 2022 only, the
eve of the opening match, then asked to simulate a tournament it has never seen. It identifies
**9 of the 16 teams that actually came out of the groups (56.2%)** — Japan topping Germany and Spain,
and Morocco's run, being exactly the kind of upset no rating-based model can anticipate.

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

Three constants in section 1 control everything: `NB_SIMULATIONS`, `SEED`, and `DATE_CUTOFF`. Setting
`DATE_CUTOFF = None` switches the model back to live mode, training on every match available today —
useful to point it at the next tournament, useless for reproducing the figures above.

## Limitations

* **Poisson independence.** Home and away goals are modelled as independent draws, which is why draws
  are under-predicted. A bivariate Poisson or a Dixon-Coles correction would address this properly.
* **No squad information.** Injuries, suspensions, and selections are invisible to the model; Elo and
  recent form are the only proxies for team strength.
* **Approximate geography.** Stadium temperatures are summer averages, not match-day forecasts, and each
  group is pinned to a single base-camp city rather than its real fixture list.
* **One tournament is one sample.** The retrospective above is encouraging, but a single World Cup
  cannot validate a probabilistic model. The match-level accuracy and the Brier score are the sturdier
  evidence.

## Author

**Sei Bayle** — personal project, built during an exchange semester at Universitas Gadjah Mada.
