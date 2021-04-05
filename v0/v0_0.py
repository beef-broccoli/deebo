# simulate bandit optimization with Suzuki dataset
# using ligand as only parameter, randomly fetch reactivity data, use average as metric
# https://ax.dev/versions/latest/tutorials/factorial.html

import random
import numpy as np
import pandas as pd
import sklearn as skl
from scipy import stats
from typing import Dict, Optional, Tuple, Union
from ax import Arm, ChoiceParameter, Models, ParameterType, SearchSpace, SimpleExperiment
from ax.plot.scatter import plot_fitted
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.stats.statstools import agresti_coull_sem
import plotly.io as pio

from suzuki_analysis import get_results

init_notebook_plotting()

ligand_names, _, ligand_results = get_results('ligand')
print(ligand_results.shape)
# ['AmPhos', 'CataCXium A', 'None', 'P(Cy)3', 'P(Ph)3 ', 'P(o-Tol)3', 'P(tBu)3', 'SPhos', 'XPhos', 'Xantphos', 'dppf', 'dtbpf']


search_space = SearchSpace(
    parameters=[
        ChoiceParameter(
            name='ligand',
            parameter_type=ParameterType.STRING,
            values=ligand_names,
        )
    ]
)

ohe = skl.preprocessing.OneHotEncoder(categories=[par.values for par in search_space.parameters.values()],
                                      sparse=False)


def evaluation_function(parameterization: Dict[str, Optional[Union[str, bool, float]]], weight: Optional[float] = None):
    batch_size = 12
    noise_level = 0
    weight = weight if weight is not None else 1.0

    features = np.array(list(parameterization.values())).reshape(1, -1)
    encoded_features = np.array(ohe.fit_transform(features))

    print(features)

    # only 1D, select all available results first, then draw samples
    names, _, results = get_results('ligand')
    results = results.reshape(len(names), -1)
    results = ((results+1)*encoded_features.T)/100
    results = list(results.ravel()[np.flatnonzero(results)])

    # print('')
    # print('results: {0}'.format(results))

    # nn = np.random.binomial(batch_size, weight)
    # n = 1 if nn == 0 else n = nn
    # print('')
    # print('{0} samples drawn'.format(n))

    samples = random.sample(results, 5)
    mean = np.average(samples)
    sem = stats.sem(samples)
    print('')
    print('mean is {0}, sem is {1}'.format(mean, sem))

    return {'success_metric': (mean, sem)}


exp = SimpleExperiment(
    name="my_experiment",
    search_space=search_space,
    evaluation_function=evaluation_function,
    objective_name="success_metric",
)

exp.status_quo = Arm(parameters={"ligand": "AmPhos"})

factorial = Models.FACTORIAL(search_space=exp.search_space)
factorial_run = factorial.gen(n=-1)  # Number of arms to generate is derived from the search space.
print('how many arms {0}'.format(len(factorial_run.arms)))

trial = (
    exp.new_batch_trial(optimize_for_power=True)
    .add_generator_run(factorial_run, multiplier=1)
)

models = []
for i in range(5):
    print(f"Running trial {i+1}...")
    data = exp.eval_trial(trial)
    thompson = Models.THOMPSON(
        experiment=exp, data=data, min_weight=0.01
    )
    models.append(thompson)
    thompson_run = thompson.gen(n=-1)
    trial = exp.new_batch_trial(optimize_for_power=True).add_generator_run(thompson_run)
