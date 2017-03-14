#!/usr/bin/env python

import numoy as np


class SVM(object):
  def __init__(self, kernel, c):
    self._kernel = kernel
    self._c = c

  def fit(self, X, y):
    """Fit the SVM using X, y

    Optimization function:
      min D(alpha) = 1/2 sum_{i,j=1..n}{yi*alphai*yj*alphaj*K(xi, xj)}
                    - sum_{i=1..n}{alphai}
    Subject to:
      0 <= alphai <= C, for all i
      sum_{i}{yi*alphai} = 0
    """
    

