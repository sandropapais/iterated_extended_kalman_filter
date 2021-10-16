import numpy as np
import sys

class IteratedExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z):

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)         # uncertainty covariance
        self.B = 0                     # control transition matrix
        self.F = np.eye(dim_x)         # state transition matrix
        self.R = np.eye(dim_z)         # state uncertainty
        self.Q = np.eye(dim_x)         # process uncertainty
        self.y = np.zeros((dim_z, 1))  # residual

        z = np.array([None]*self.dim_z)
        self.z = np.reshape_z(z, self.dim_z, self.x.ndim)

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape) # kalman gain
        self.y = np.zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self._log_likelihood = np.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def iekf_filter(pos_meas, vel_meas, pos_true, t, samples_count, t_step, q_proc_noise_cov, r_meas_noise_cov,
                     flg_debug_prop_only, flg_debug_fwd_only, update_interval_indices, a_trans_mat, c_obs_mat):
        # RTS smoother initialization
        pos_est_ini = pos_meas[0]
        pos_var_ini = q_proc_noise_cov
        pos_est_pred_fwd = np.zeros(samples_count)
        pos_est_corr_fwd = np.zeros(samples_count)
        pos_est_corr = np.zeros(samples_count)
        pos_var_pred_fwd = np.zeros(samples_count)
        pos_var_corr_fwd = np.zeros(samples_count)
        pos_var_corr = np.zeros(samples_count)

        # RTS smoother forward pass
        for i in range(0, samples_count):
            if i == 0:
                pos_est_pred_fwd[0] = pos_est_ini
                pos_var_pred_fwd[0] = pos_var_ini
            else:
                pos_var_pred_fwd[i] = a_trans_mat * pos_var_corr_fwd[i - 1] * a_trans_mat + q_proc_noise_cov
                pos_est_pred_fwd[i] = a_trans_mat * pos_est_corr_fwd[i - 1] + t_step * vel_meas[i]
            if flg_debug_prop_only == 1:
                kalman_gain = 0
            elif (i % update_interval_indices) == 0:
                kalman_gain = pos_var_pred_fwd[i] * c_obs_mat / (
                        c_obs_mat * pos_var_pred_fwd[i] * c_obs_mat + r_meas_noise_cov)
            else:
                kalman_gain = 0
            pos_var_corr_fwd[i] = (1 - kalman_gain * c_obs_mat) * pos_var_pred_fwd[i]
            pos_est_corr_fwd[i] = pos_est_pred_fwd[i] + kalman_gain * (pos_meas[i] - c_obs_mat * pos_est_pred_fwd[i])

        # RTS smoother backward pass
        if flg_debug_fwd_only == 1:
            pos_var_corr = pos_var_corr_fwd
            pos_est_corr = pos_est_corr_fwd
        else:
            pos_est_corr[-1] = pos_est_corr_fwd[-1]
            pos_var_corr[-1] = pos_var_corr_fwd[-1]
            for i in range(samples_count - 1, 0, -1):
                pos_est_corr[i - 1] = \
                    pos_est_corr_fwd[i - 1] + (pos_var_corr_fwd[i - 1] * a_trans_mat / pos_var_pred_fwd[i]) * \
                    (pos_est_corr[i] - pos_est_pred_fwd[i])
                pos_var_corr[i - 1] = \
                    pos_var_corr_fwd[i - 1] + (pos_var_corr_fwd[i - 1] * a_trans_mat / pos_var_pred_fwd[i]) * \
                    (pos_var_corr[i] - pos_var_pred_fwd[i]) * (pos_var_corr_fwd[i - 1] * a_trans_mat / pos_var_pred_fwd[i])