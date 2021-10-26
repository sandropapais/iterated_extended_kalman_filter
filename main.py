from os.path import dirname, join
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# import iterated_extended_kalman_filter as iekf
from sympy import latex


def read_data_mat(subdir_name, file_name):
    # Load .mat file
    mat_file_dir = join(dirname(__file__), subdir_name)
    mat_file_path = join(mat_file_dir, file_name)
    data_dict = sio.loadmat(mat_file_path)
    return data_dict


def unpack_data(data_dict):
    # Unpack data and flatten to 1D
    t = data_dict['t'].flatten()
    range_meas = data_dict['r']
    ang_meas = data_dict['b']
    vel_norm_meas = data_dict['v'].flatten()
    ang_vel_meas = data_dict['om'].flatten()
    pos_mat_landmarks_true = data_dict['l']
    landmarks_pos_x_true = pos_mat_landmarks_true[:, 0]
    landmarks_pos_y_true = pos_mat_landmarks_true[:, 1]
    pos_x_true = data_dict['x_true'].flatten()
    pos_y_true = data_dict['y_true'].flatten()
    ang_true = data_dict['th_true'].flatten()
    valid_flg = data_dict['true_valid'].flatten()
    dist_lidar_rel_body = data_dict['d'].flatten()
    range_meas_var = data_dict['r_var'].flatten()  # unused
    ang_meas_var = data_dict['b_var'].flatten()  # unused
    vel_meas_var = data_dict['v_var'].flatten()  # unused
    ang_vel_meas_var = data_dict['om_var'].flatten()  # unused
    t_step_count = t.size
    landmarks_count = landmarks_pos_x_true.size
    t_step = 0.1

    print(f'Provided meas variance values: range={range_meas_var}, vel={vel_meas_var}, ang={ang_meas_var}, angvel={ang_vel_meas_var}')

    return t, range_meas, ang_meas, vel_norm_meas, ang_vel_meas, landmarks_pos_x_true, landmarks_pos_y_true, pos_x_true, \
           pos_y_true, ang_true, valid_flg, dist_lidar_rel_body, t_step_count, t_step, landmarks_count


def plot_data(t, range_meas, ang_meas, vel_norm_meas, ang_vel_meas, pos_x_true, pos_y_true, ang_true, t_step, valid_flg,
              landmarks_pos_x_true, landmarks_pos_y_true, dist_lidar_rel_body):
    # PLT01: Laser Measurements
    landmark_id = 0
    range_true = np.sqrt((landmarks_pos_x_true[landmark_id] - pos_x_true - dist_lidar_rel_body * np.cos(ang_true)) ** 2
                         + (landmarks_pos_y_true[landmark_id] - pos_y_true - dist_lidar_rel_body * np.sin(
        ang_true)) ** 2)
    ldr_ang_true = np.arctan2(landmarks_pos_y_true[landmark_id] - pos_y_true - dist_lidar_rel_body * np.sin(ang_true),
                              landmarks_pos_x_true[landmark_id] - pos_x_true - dist_lidar_rel_body * np.cos(ang_true)) \
                   - ang_true
    range_meas_err = range_true - range_meas[:, landmark_id]
    ang_meas_err = ldr_ang_true - ang_meas[:, landmark_id]
    range_meas_err = np.where(range_meas[:, landmark_id] == 0, 0, range_meas_err)
    ang_meas_err = np.where(range_meas[:, landmark_id] == 0, 0, ang_meas_err)
    ang_meas_err = np.where(ang_meas_err > np.pi, ang_meas_err - 2 * np.pi, ang_meas_err)
    ang_meas_err = np.where(ang_meas_err < -np.pi, ang_meas_err + 2 * np.pi, ang_meas_err)
    fig1, axs1 = plt.subplots(2, 2)
    axs1[0, 0].plot(t, range_meas[:, landmark_id])
    axs1[0, 0].set_ylabel('Laser Range (m)')
    axs1[0, 0].set_title('Laser Measurement')
    axs1[1, 0].plot(t, ang_meas[:, landmark_id])
    axs1[1, 0].set_xlabel('Time (s)')
    axs1[1, 0].set_ylabel('Laser Bearing (rad)')
    axs1[0, 1].plot(t, range_meas_err)
    axs1[0, 1].set_title('Laser Measurement Error')
    axs1[1, 1].plot(t, ang_meas_err)
    axs1[1, 1].set_xlabel('Time (s)')
    fig1.tight_layout(pad=0.2)
    fig1.savefig('out/laser_vs_t.png')

    # PLT02: Odometery Measurements
    vel_x_true = (pos_x_true[1:] - pos_x_true[:-1]) / t_step
    vel_y_true = (pos_y_true[1:] - pos_y_true[:-1]) / t_step
    vel_norm_true = np.sqrt(vel_x_true ** 2 + vel_y_true ** 2)
    vel_norm_true = np.insert(vel_norm_true, 0, vel_norm_true[0])
    vel_norm_meas_err = vel_norm_true - vel_norm_meas
    ang_vel_true = (ang_true[1:] - ang_true[:-1]) / t_step
    ang_vel_true = np.insert(ang_vel_true, 0, ang_vel_true[0])
    ang_vel_true = np.where(ang_vel_true > np.pi / t_step, ang_vel_true - 2 * np.pi / t_step, ang_vel_true)
    ang_vel_true = np.where(ang_vel_true < -np.pi / t_step, ang_vel_true + 2 * np.pi / t_step, ang_vel_true)
    ang_vel_meas_err = ang_vel_true - ang_vel_meas
    fig2, axs2 = plt.subplots(2, 2)
    axs2[0, 0].plot(t, vel_norm_meas, label='meas')
    axs2[0, 0].plot(t, vel_norm_true, label='true')
    axs2[0, 0].set_ylabel('Odometery Speed (m/s)')
    axs2[0, 0].legend()
    axs2[0, 0].set_title('Odometery Measurements')
    axs2[1, 0].plot(t, ang_vel_meas, label='meas')
    axs2[1, 0].plot(t, ang_vel_true, label='true')
    axs2[1, 0].legend()
    axs2[1, 0].set_xlabel('Time (s)')
    axs2[1, 0].set_ylabel('Odometery Angular Rate (rad/s)')
    axs2[0, 1].plot(t, vel_norm_meas_err, label='meas')
    axs2[0, 1].set_title('Odometery Error')
    axs2[1, 1].plot(t, ang_vel_meas_err, label='meas')
    axs2[1, 1].set_xlabel('Time (s)')
    fig2.savefig('out/odom_vs_t.png')

    # PLT03: True States
    fig3, axs3 = plt.subplots(3, 1)
    axs3[0].plot(t, pos_x_true)
    axs3[0].set_ylabel('Pos x (m)')
    axs3[0].set_title('True States')
    axs3[1].plot(t, pos_y_true)
    axs3[1].set_ylabel('Pos y (rad)')
    axs3[2].plot(t, ang_true)
    axs3[2].set_xlabel('Time (s)')
    axs3[2].set_ylabel('Angle (rad)')
    fig3.savefig('out/true_vs_t.png')

    # PLT04: Valid States
    # fig4, axs4 = plt.subplots()
    # axs4.plot(t, valid_flg)
    # axs4.set_ylabel('Valid Flag')
    # axs4.set_title('True State Validity')
    # axs4.set_xlabel('Time (s)')


def compute_measurement_stats(flg_print_stats, range_meas, ang_meas, vel_norm_meas, ang_vel_meas, landmarks_pos_x_true,
                              landmarks_pos_y_true, pos_x_true, pos_y_true, ang_true, valid_flg, dist_lidar_rel_body,
                              t_step_count, t_step, landmarks_count):
    # Speed process noise
    proc_noise = np.zeros((2, t_step_count))
    proc_noise[0, 1:] = np.sqrt((pos_x_true[1:] - pos_x_true[:-1]) ** 2 + (pos_y_true[1:] - pos_y_true[:-1]) ** 2) \
                        / t_step - vel_norm_meas[1:]
    # Angular rate process noise
    ang_true_del = ang_true[1:] - ang_true[:-1]
    proc_noise[1, 1:] = ang_true_del / t_step - ang_vel_meas[1:]
    # Angle wrapping
    proc_noise[1, 1:] = np.where(ang_true_del > np.pi,
                                 (ang_true_del - 2 * np.pi) / t_step - ang_vel_meas[1:],
                                 proc_noise[1, 1:])
    proc_noise[1, 1:] = np.where(ang_true_del < -np.pi,
                                 (ang_true_del + 2 * np.pi) / t_step - ang_vel_meas[1:],
                                 proc_noise[1, 1:])
    # Transformed process noise, w_k^\prime
    proc_noise_transform = np.zeros((3, t_step_count))
    proc_noise_transform[0, :] = proc_noise[0, :]*t_step*np.cos(ang_true)
    proc_noise_transform[1, :] = proc_noise[0, :]*t_step*np.sin(ang_true)
    proc_noise_transform[2, :] = proc_noise[1, :]*t_step

    # Measurement noise
    meas_count = np.count_nonzero(range_meas)
    meas_noise = np.zeros((2, meas_count))  # motion model noise
    k = 0
    for i in range(t_step_count):
        for j in range(landmarks_count):
            if range_meas[i, j] != 0:
                meas_noise[0, k] = \
                    range_meas[i, j] - \
                    np.sqrt(
                        (landmarks_pos_x_true[j] - pos_x_true[i] - dist_lidar_rel_body * np.cos(ang_true[i])) ** 2 +
                        (landmarks_pos_y_true[j] - pos_y_true[i] - dist_lidar_rel_body * np.sin(ang_true[i])) ** 2)
                meas_noise[1, k] = \
                    ang_meas[i, j] + ang_true[i] - \
                    np.arctan2(landmarks_pos_y_true[j] - pos_y_true[i] - dist_lidar_rel_body * np.sin(ang_true[i]),
                               landmarks_pos_x_true[j] - pos_x_true[i] - dist_lidar_rel_body * np.cos(ang_true[i]))
                k = k + 1
    # Angle wrapping
    meas_noise = np.where(meas_noise > np.pi, meas_noise - 2 * np.pi, meas_noise)
    meas_noise = np.where(meas_noise < -np.pi, meas_noise + 2 * np.pi, meas_noise)

    # Compute noise statistics
    pos_true_mean = np.mean(np.abs(pos_x_true) * np.abs(pos_y_true))
    vel_norm_meas_mean = np.mean(np.abs(vel_norm_meas))
    ang_vel_meas_mean = np.mean(np.abs(ang_vel_meas))
    meas_noise_mean = np.mean(meas_noise, axis=1)
    meas_noise_var = np.var(meas_noise, axis=1)
    r_meas_noise_cov = np.cov(meas_noise)
    proc_noise_mean = np.mean(proc_noise, axis=1)
    proc_noise_var = np.var(proc_noise, axis=1)
    q_proc_noise_cov = np.cov(proc_noise)
    proc_noise_transform_var = np.var(proc_noise_transform, axis=1)
    q_proc_noise_transform_cov = np.cov(proc_noise_transform)

    # Noise statistics print
    if flg_print_stats == 1:
        print('*** Noise Statistics  ***\n'
              f'Valid data count is {np.sum(valid_flg)} out of {t_step_count}\n'
              f'Measurement count is {meas_count}, last count is {k}\n'
              f'Position mean = {pos_true_mean:1.3f} m, speed mean = {vel_norm_meas_mean:1.3f} m/s, '
              f'ang vel mean = {ang_vel_meas_mean:1.3f} rad/s\n'
              f'Measurement noise: mean={meas_noise_mean}, 3std={3*np.sqrt(meas_noise_var)},\n'
              f' std={np.sqrt(meas_noise_var)}, cov = {r_meas_noise_cov}\n'
              f'Process noise: mean={proc_noise_mean}, 3std={3*np.sqrt(proc_noise_var)},\n'
              f' std={np.sqrt(proc_noise_var)}, cov = {q_proc_noise_cov}\n'
              f'Process noise transformed: 3std={3*np.sqrt(proc_noise_transform_var)},\n'
              f' std={np.sqrt(proc_noise_transform_var)}, cov = {q_proc_noise_transform_cov}\n')

    return proc_noise, meas_noise, pos_true_mean, meas_noise_mean, r_meas_noise_cov, proc_noise_mean, \
           q_proc_noise_transform_cov


def plot_noise(t, proc_noise, meas_noise, meas_noise_mean, r_meas_noise_cov, proc_noise_mean, q_proc_noise_cov):
    # PLT02: Model Noise
    fig2, axs2 = plt.subplots(2, 2)
    axs2[0, 0].plot(t, proc_noise[0, :])
    axs2[0, 0].set_ylabel('Speed (m/s)')
    axs2[0, 0].set_title("Process Noise")
    axs2[1, 0].plot(t, proc_noise[1, :])
    axs2[1, 0].set_xlabel('Time (s)')
    axs2[1, 0].set_ylabel('Angular Rate (rad/s)')
    axs2[0, 1].plot(np.linspace(t[0], t[-1], meas_noise[0, :].size), meas_noise[0, :])
    axs2[0, 1].set_ylabel('Range (m)')
    axs2[0, 1].set_title("Measurement Noise")
    axs2[1, 1].plot(np.linspace(t[0], t[-1], meas_noise[1, :].size), meas_noise[1, :])
    axs2[1, 1].set_xlabel('Time (s)')
    axs2[1, 1].set_ylabel('Bearing (rad)')
    fig2.tight_layout(pad=0.2)
    fig2.savefig('out/model_noise_vs_t.png')

    # PLOT03: Model Noise Histogram
    fig3, axs3 = plt.subplots(2, 2)
    n_pts_plot = 100
    bins = axs3[0, 0].hist(proc_noise[0, :], 30, range=(-0.2, 0.2),density=1)[1]
    proc_noise_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
    proc_noise_std = np.sqrt(q_proc_noise_cov[0, 0])
    proc_noise_fit = norm.pdf(proc_noise_fit_bins, proc_noise_mean[0], proc_noise_std)
    axs3[0, 0].plot(proc_noise_fit_bins, proc_noise_fit)
    axs3[0, 0].set_xlabel('Speed Process Noise (m/s)')
    axs3[0, 0].text(0.95, 0.95, fr'$\mu={proc_noise_mean[0]:1.3e}, \sigma={proc_noise_std:1.3f}$',
                    horizontalalignment='right', verticalalignment='top', transform=axs3[0, 0].transAxes)
    axs3[0, 0].set_title("Process Noise Histogram")

    bins = axs3[1, 0].hist(proc_noise[1, :], 30, range=(-0.3, 0.3), density=1)[1]
    proc_noise_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
    proc_noise_std = np.sqrt(q_proc_noise_cov[1, 1])
    proc_noise_fit = norm.pdf(proc_noise_fit_bins, proc_noise_mean[1], proc_noise_std)
    axs3[1, 0].plot(proc_noise_fit_bins, proc_noise_fit)
    axs3[1, 0].set_xlabel('Angular Rate Process Noise (m/s)')
    axs3[1, 0].text(0.95, 0.95, fr'$\mu={proc_noise_mean[1]:1.3e}, \sigma={proc_noise_std:1.3f}$',
                    horizontalalignment='right', verticalalignment='top', transform=axs3[1, 0].transAxes)

    bins = axs3[0, 1].hist(meas_noise[0, :], 30, range=(-0.15, 0.15), density=1)[1]
    meas_noise_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
    meas_noise_std = np.sqrt(r_meas_noise_cov[0, 0])
    meas_noise_fit = norm.pdf(meas_noise_fit_bins, meas_noise_mean[0], meas_noise_std)
    axs3[0, 1].plot(meas_noise_fit_bins, meas_noise_fit)
    axs3[0, 1].set_xlabel('Range Measurement Noise (m)')
    axs3[0, 1].text(0.95, 0.95, fr'$\mu={meas_noise_mean[0]:1.3e}, \sigma={meas_noise_std:1.3f}$',
                    horizontalalignment='right', verticalalignment='top', transform=axs3[0, 1].transAxes)
    axs3[0, 1].set_title("Measurement Noise Histogram")

    bins = axs3[1, 1].hist(meas_noise[1, :], 30, range=(-0.1, 0.1), density=1)[1]
    meas_noise_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
    meas_noise_std = np.sqrt(r_meas_noise_cov[1, 1])
    meas_noise_fit = norm.pdf(meas_noise_fit_bins, meas_noise_mean[1], meas_noise_std)
    axs3[1, 1].plot(meas_noise_fit_bins, meas_noise_fit)
    axs3[1, 1].set_xlabel('Bearing Measurement Noise (rad)')
    axs3[1, 1].text(0.95, 0.95, fr'$\mu={meas_noise_mean[1]:1.3e}, \sigma={meas_noise_std:1.3f}$',
                    horizontalalignment='right', verticalalignment='top', transform=axs3[1, 1].transAxes)
    fig3.tight_layout(pad=0.2)
    fig3.savefig('out/hist_model_noise.png')

    # PLOT04: Model Noise Q-Q
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(2, 2, 1)
    qqplot(proc_noise[0, :], ax=ax4, line='s')
    ax4.set_ylabel('Speed\nProcess Noise')
    ax4 = fig4.add_subplot(2, 2, 3)
    qqplot(proc_noise[1, :], ax=ax4, line='s')
    ax4.set_ylabel('Angular Rate\nProcess Noise')
    ax4 = fig4.add_subplot(2, 2, 2)
    qqplot(meas_noise[0, :], ax=ax4, line='s')
    ax4.set_ylabel('Range\nMeasurement Noise')
    ax4 = fig4.add_subplot(2, 2, 4)
    qqplot(meas_noise[1, :], ax=ax4, line='s')
    ax4.set_ylabel('Bearing\nMeasurement Noise')
    fig4.suptitle("Model Noise Q-Q")
    fig4.tight_layout(pad=0.3)
    fig4.savefig('out/qq_model_noise.png')

    # PLT05: Model Noise Covariance
    fig5, axs5 = plt.subplots(1, 2)
    axs5[0].scatter(proc_noise[0, :], proc_noise[1, :], s=0.5)
    axs5[0].set_xlabel('Speed Process Noise (m/s)')
    axs5[0].set_ylabel("Angular Rate Process Noise (rad/s)")
    confidence_ellipse(proc_noise[0, :], proc_noise[1, :], axs5[0], n_std=3, label=r'$3\sigma$', edgecolor='tab:orange')
    axs5[0].legend()
    axs5[1].scatter(meas_noise[0, :], meas_noise[1, :], s=0.5)
    axs5[1].set_xlabel('Range Meas Noise (m/s)')
    axs5[1].set_ylabel("Bearing Meas Noise (rad/s)")
    confidence_ellipse(meas_noise[0, :], meas_noise[1, :], axs5[1], n_std=3, label=r'$3\sigma$', edgecolor='tab:orange')
    axs5[1].legend()
    fig5.suptitle("Model Noise Covariance")
    fig5.tight_layout(pad=0.3)
    fig5.savefig('out/cov_model_noise.png')


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def print_jacobians():
    import sympy
    sympy.init_printing(use_latex='mathjax')

    x, y, theta, T, v, w1, w2, r, psi, n1, n2, xl, yl, d, om = sympy.symbols('x,y,theta,T,v,w1,w2,r,psi,n1,n2,xl,yl,d,om')
    state = sympy.Matrix([x, y, theta])
    motion_model = sympy.Matrix([x+T*sympy.cos(theta)*(v+w1), y+T*sympy.sin(theta)*(v+w1), theta+T*(om+w2)])
    motion_model_jacobian = motion_model.jacobian(state)
    measurement_model = sympy.Matrix([sympy.sqrt((xl-x-d*sympy.cos(theta))**2+(yl-y-d*sympy.sin(theta))**2)+n1,
                                      sympy.atan2(yl-y-d*sympy.sin(theta), xl-x-d*sympy.cos(theta))-theta+n2])
    measurement_model_jacobian = measurement_model.jacobian(state)

    print('F = ', latex(motion_model_jacobian))
    print('G = ', latex(measurement_model_jacobian))


# def plot_iekf():
#     # Post process results
#     pos_est_err = pos_est_corr - pos_true
#     pos_est_err_mean = np.mean(pos_est_err)
#     pos_est_err_mod_avg = np.mean(np.abs(pos_est_err))
#     pos_est_rmse = np.sqrt(np.sum(pos_est_err ** 2) / samples_count)
#     pos_est_err_std = np.std(pos_est_err)
#     pos_3sigma = 3 * np.sqrt(pos_var_corr)
#     print('*** POST-PROCESS ****\n'
#           f'Process model variance = {q_proc_noise_cov:1.5f}\n'
#           f'Measurement model variance = {r_meas_noise_cov:1.5f}\n'
#           f'Average error magnitude = {pos_est_err_mod_avg:1.3f} m\n'
#           f'Root mean square error = {pos_est_rmse:1.3f} m\n'
#           f'3 sigma error = {3 * pos_est_err_std:1.3f} m')
#
#     # PLOT05: Estimation Error and Uncertainty
#     fig5, ax5 = plt.subplots()
#     ax5.plot(t, pos_est_err, label=r'$\hat x_k - x_k$')
#     ax5.plot(t, pos_3sigma, 'r--', label=r'$\pm3\hat\sigma_{x_k}$')
#     ax5.plot(t, -pos_3sigma, 'r--')
#     ax5.set_xlabel('Time (s)')
#     ax5.set_ylabel('Position (m)')
#     ax5.set_title("Estimation Error and Uncertainty")
#     ax5.legend()
#     fig5.savefig(f'out/est_err_and_3std_{update_interval_indices}steps.png')
#
#     # PLOT06: Estimation Error Histogram
#     fig6, ax6 = plt.subplots()
#     n_pts_plot = 1000
#     bins = ax6.hist(pos_est_err, 20, density=1)[1]
#     pos_est_err_fit_bins = np.linspace(bins[0], bins[-1], n_pts_plot)
#     pos_est_err_fit = norm.pdf(pos_est_err_fit_bins, 0, pos_est_err_std)
#     ax6.plot(pos_est_err_fit_bins, pos_est_err_fit)
#     ax6.set_xlabel('Position Estimate Error (m)')
#     ax6.set_title("Estimation Error Histogram")
#     plt.text(0.95, 0.95, fr'$\mu={pos_est_err_mean:1.3e}, \sigma={pos_est_err_std:1.3f}$',
#              horizontalalignment='right', verticalalignment='top', transform=ax6.transAxes)
#     fig6.savefig(f'out/est_err_hist_{update_interval_indices}steps.png')
#
#     # PLOT07: Estimation and Uncertainty
#     fig7, ax7 = plt.subplots()
#     ax7.plot(t, pos_est_corr, label=r'$\hat{x_k}$')
#     ax7.plot(t, pos_true, label=r'$x_k$')
#     ax7.plot(t, pos_est_corr + pos_3sigma, 'r--', label=r'$\hat{x_k}\pm3\hat\sigma_{x_k}$')
#     ax7.plot(t, pos_est_corr - pos_3sigma, 'r--')
#     ax7.set_xlabel('Time (s)')
#     ax7.set_ylabel('Position (m)')
#     ax7.set_title("Estimation and Uncertainty")
#     ax7.legend()
#     fig7.savefig(f'out/est_and_tru_{update_interval_indices}steps.png')

def main():
    # Define flags
    flg_plot_show = 0  # 0, 1
    flg_print_stats = 1  # 0, 1
    flg_print_jacobians = 0  # 0, 1
    flg_debug_prop_only = 0  # 0, 1
    # Define parameters
    range_max = 1  # 1, 3, 5
    a_trans_mat = 1
    c_obs_mat = 1
    # Load data file
    subdir_name = 'data'
    file_name = 'dataset2.mat'
    data_dict = read_data_mat(subdir_name, file_name)
    t, range_meas, ang_meas, vel_norm_meas, ang_vel_meas, landmarks_pos_x_true, landmarks_pos_y_true, pos_x_true, \
    pos_y_true, ang_true, valid_flg, dist_lidar_rel_body, t_step_count, t_step, landmarks_count \
        = unpack_data(data_dict)
    plot_data(t, range_meas, ang_meas, vel_norm_meas, ang_vel_meas, pos_x_true, pos_y_true, ang_true, t_step, valid_flg,
              landmarks_pos_x_true, landmarks_pos_y_true, dist_lidar_rel_body)
    # Q1: Compute statistics from measurement data
    proc_noise, meas_noise, pos_true_mean, meas_noise_mean, r_meas_noise_cov, proc_noise_mean, q_proc_noise_cov = \
        compute_measurement_stats(flg_print_stats, range_meas, ang_meas, vel_norm_meas, ang_vel_meas,
                                  landmarks_pos_x_true, landmarks_pos_y_true, pos_x_true, pos_y_true, ang_true,
                                  valid_flg, dist_lidar_rel_body, t_step_count, t_step, landmarks_count)
    plot_noise(t, proc_noise, meas_noise, meas_noise_mean, r_meas_noise_cov, proc_noise_mean, q_proc_noise_cov)
    # Q2 Verify Jacobians derivation
    if flg_print_jacobians == 1:
        print_jacobians()
    # Q4 Call EKF on data

    # Q5: Call RTS smoother on data
    # q_proc_noise_cov = 0.002  # instead of using prop_err_std ** 2, we inflate the process noise
    # iekf.iekf_filter(pos_meas, vel_meas, pos_true, t, samples_count, t_step, q_proc_noise_cov, r_meas_noise_cov,
    #                  flg_debug_fwd_only, update_interval_indices, a_trans_mat, c_obs_mat)
    # Plotting
    if flg_plot_show == 1:
        plt.show()


if __name__ == "__main__":
    main()
