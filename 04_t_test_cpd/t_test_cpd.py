from typing import Tuple
import numpy as np
from scipy import stats


class TTestCPDetector:
    def __init__(self, t_warmup=30, p_limit=0.01) -> None:
        self._t_warmup = t_warmup
        self._p_limit = p_limit
        self.current_t = 0
        self.current_obs = []
        self.population_mean = None
        self.current_std = None

    def predict_next(self, y) -> Tuple[float, bool]:
        self._update_data(y)
        if self.current_t == self._t_warmup:  # if the number of collected observations fills the warmup window, calcuate population mean.
            self._init_params()
        if self.current_t > self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()
            return (1-prob), is_changepoint
        return 0, False

    def _reset(self) -> None:
        self.current_t = 0
        self.current_obs = []
        self.population_mean = None
        self.current_std = None

    def _update_data(self, y) -> None:
        self.current_t += 1
        self.current_obs.append(y.reshape(1))

    def _init_params(self) -> None:
        self.population_mean = np.mean(np.array(self.current_obs))

    def _check_for_changepoint(self) -> Tuple[float, bool]:
        current_std = None
        if self._t_warmup == 1:  # prevent zero division when calculating sample std.
            current_std = 1e-3
        else:
            current_std = np.sqrt(1.0/(self.current_t-1)*np.sum(np.power(np.array(self.current_obs)-np.mean(np.array(self.current_obs)), 2)))
        test_statistic = (np.mean(np.array(self.current_obs)) - self.population_mean)/(current_std / (self.current_t**0.5))
        prob = float(self._get_prob(test_statistic))
        return prob, prob < self._p_limit

    def _get_prob(self, y) -> bool:
        p = stats.t.cdf(np.abs(y), self._t_warmup)
        prob = 2*(1 - p)
        return prob

    def get_t_warmup(self) -> int:
        return self._t_warmup

    def set_t_warmup(self, t_warmup: int) -> None:
        self._t_warmup = t_warmup
