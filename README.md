# ppiplan

Plan the human sample of a prediction-powered (mixed-subjects) study from a small pilot.

Studies that combine human answers with language model predictions size their human sample from the pilot correlation between predictions and answers. Choosing the model, prompt or persona format with the highest pilot correlation and then planning with that same correlation makes the plan too optimistic: in our data the delivered correlation is about 0.12 lower and realized power falls from 0.80 to about 0.73. `ppiplan` plans with the average of standardized predictions from configurations fixed before the pilot, so the pilot is never used to choose, and it reports the power a study would have if the true correlation were lower.

## Install

```
pip install ppiplan
```

## Use

```python
from ppiplan import plan

p = plan(y_pilot, predictions_pilot, effect=0.2)
p.n_human
p.share_saved
p.power_if_rho_is
```

`predictions_pilot` is an array with one column per configuration fixed in advance. For a two-arm experiment, pass `arm` to plan for the treatment effect, and pass `lower_bound=True` for a conservative plan. `python examples/simulate.py` reproduces the comparison between the average and choosing on the pilot.

## Reference

Crainic, J., and Yee, B. Choosing a Language Model on the Pilot Overstates the Savings of Mixed-Subjects Designs. Extended abstract, 2026.

## License

MIT
