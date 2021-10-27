from copy import deepcopy
import numpy as np
from scipy import linalg


class IteratedExtendedKalmanFilter(object):

    def __init__(self, dim_state, dim_meas):

        self.dim_x = dim_state
        self.dim_meas = dim_meas

        self.x = np.zeros((dim_state, 1))     # state
        self.P = np.eye(dim_state)            # state uncertainty covariance
        self.F = np.eye(dim_state)            # state transition matrix
        self.R = np.eye(dim_meas)             # measurement noise covariance
        self.Q = np.eye(dim_state)            # process noise covariance
        self.y_res = np.zeros((dim_meas, 1))  # measurement residual
        self.y = np.zeros((dim_meas, 1))      # measurement

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape)  # kalman gain
        self.S = np.zeros((dim_meas, dim_meas))   # system uncertainty
        self.S_inv = np.zeros((dim_meas, dim_meas))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_state)

        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, f, f_args=()):
        """
        Predict next state (prior) using the Kalman filter state propagation equations.
        Parameters
        ----------
        f : function
            function which takes as input the past state variable (self.x) along
            with the optional arguments in f_args, and returns the predicted state.
        f_args : tuple, optional, default (,)
            arguments to be passed into f after the required state variable.
        """

        if not isinstance(f_args, tuple):
            f_args = (f_args,)

        self.x = f(self.x, *f_args)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, y, G_fun, g_fun, R=None, G_args=(), g_args=(),
               residual=np.subtract):
        """ Performs the update innovation of the extended Kalman filter.
        Parameters
        ----------
        y : np.array
            measurement for this step.
            If `None`, posterior is not computed
        G_fun : function
           function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, returns H.
        g_fun : function
            function which takes as input the state variable (self.x) along
            with the optional arguments in hx_args, and returns the measurement
            that would correspond to that state.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        G_args : tuple, optional, default (,)
            arguments to be passed into HJacobian after the required state
            variable. for robot localization you might need to pass in
            information about the map and time of day, so you might have
            `args=(map_data, time)`, where the signature of HCacobian will
            be `def HJacobian(x, map, t)`
        g_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state
            variable.
        residual : function (z, z2), optional
            Optional function that computes the residual (difference) between
            the two measurement vectors. If you do not provide this, then the
            built in minus operator will be used. You will normally want to use
            the built in unless your residual computation is nonlinear (for
            example, if they are angles)
        """

        if y is None:
            self.y = np.array([[None] * self.dim_meas]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(G_args, tuple):
            G_args = (G_args,)

        if not isinstance(g_args, tuple):
            g_args = (g_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_meas) * R

        if np.isscalar(y) and self.dim_meas == 1:
            y = np.asarray([y], float)

        G = G_fun(self.x, *G_args)

        # Kalman gain computation
        PH_trans = np.dot(self.P, G.T)
        self.S = np.dot(G, PH_trans) + R
        self.S_inv = linalg.inv(self.S)
        self.K = PH_trans.dot(self.S_inv)

        # State update
        g = g_fun(self.x, *g_args)
        self.y_res = residual(y, g)
        self.x = self.x + np.dot(self.K, self.y_res)

        # P = (I-KG)P(I-KG)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KG)P usually seen in the literature.
        I_minus_KG = self._I - np.dot(self.K, G)
        self.P = np.dot(I_minus_KG, self.P).dot(I_minus_KG.T) + np.dot(self.K, R).dot(self.K.T)

        # set to None to force recompute
        self._mahalanobis = None

        # save measurement and posterior state
        self.y = deepcopy(y)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    @property
    def mahalanobis(self):
        """
        Mahalanobis distance of innovation. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.
        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(np.dot(np.dot(self.y.T, self.S_inv), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'KalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('S', self.S),
            pretty_str('mahalanobis', self.mahalanobis)
            ])


def pretty_str(label, arr):
    """
    Generates a pretty printed NumPy array with an assignment. Optionally
    transposes column vectors so they are drawn on one line. Strictly speaking
    arr can be any time convertible by `str(arr)`, but the output may not
    be what you want if the type of the variable is not a scalar or an
    ndarray.
    Examples
    --------
    pprint('cov', np.array([[4., .1], [.1, 5]]))
    >> cov = [[4.  0.1]
           [0.1 5. ]]
    >> print(pretty_str('x', np.array([[1], [2], [3]])))
    x = [[1 2 3]].T
    """

    def is_col(a):
        """ return true if a is a column vector"""
        try:
            return a.shape[0] > 1 and a.shape[1] == 1
        except (AttributeError, IndexError):
            return False

    if label is None:
        label = ''

    if label:
        label += ' = '

    if is_col(arr):
        return label + str(arr.T).replace('\n', '') + '.T'

    rows = str(arr).split('\n')
    if not rows:
        return ''

    s = label + rows[0]
    pad = ' ' * len(label)
    for line in rows[1:]:
        s = s + '\n' + pad + line

    return s