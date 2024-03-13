import numpy as np
import scipy
from typing import Tuple
from scipy import stats


class CusumDetector:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.s = 0

    def _reset(self):
        self.s = 0

    def detect(self, observation: float) -> bool:
        is_change = False
        self.s = max(0, self.s+observation-self.threshold)
        if self.s > self.threshold:
            is_change = True
        if is_change:
            self._reset()
        return is_change


class CUSUMProbabilityDetector:
    def __init__(self, t_warmup=30, p_limit=0.01) -> None:
        self._t_warmup = t_warmup
        self._p_limit = p_limit
        self.current_t = 0
        self.current_obs = []
        self.current_mean = None
        self.current_std = None

    def predict_next(self, y: float) -> Tuple[float, bool]:
        self._update_data(y)
        if self.current_t == self._t_warmup:
            self._init_params()
        if self.current_t >= self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()
            return (1-prob), is_changepoint
        else:
            return 0, False

    def _reset(self) -> None:
        self.current_t = 0
        self.current_obs = []
        self.current_mean = None
        self.current_std = None

    def _update_data(self, y: float) -> None:
        self.current_t += 1
        self.current_obs.append(y)

    def _init_params(self) -> None:
        self.current_mean = np.mean(self.current_obs)
        self.current_std = np.std(self.current_obs)

    def _check_for_changepoint(self) -> Tuple[float, bool]:
        standardized_sum = np.sum(np.array(self.current_obs) - self.current_mean)/(self.current_std * self.current_t**0.5)
        prob = float(self._get_prob(standardized_sum))
        return prob, prob < self._p_limit

    def _get_prob(self, y: float) -> bool:
        p = stats.norm.cdf(y)
        prob = 2*(1 - p)
        return prob
