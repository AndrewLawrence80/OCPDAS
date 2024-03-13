"""
Copy from [Ruptures](https://centre-borelli.github.io/ruptures-docs/) 
"""
from ..utils import pairwise


class BaseCost:
    def fit(self, *args, **kwargs):
        """Set the parameters of the cost function, for instance the Gram matrix, etc."""
        pass

    def error(self, start, end):
        """Returns the cost on segment [start:end]."""
        pass

    def sum_of_costs(self, bkps):
        """Returns the sum of segments cost for the given segmentation.

        Args:
            bkps (list): list of change points. By convention, bkps[-1]==n_samples.

        Returns:
            float: sum of costs
        """
        sum_of_cost = sum(self.error(start, end) for start, end in pairwise([0] + bkps))
        return sum_of_cost

    def model(self):
        pass

    def get_min_size(self):
        """Returns the minimum size of the segment size which the cost model will apply to"""
        pass
