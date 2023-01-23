""" A statistical approach to improve the places on the map"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from matplotlib.patches import Ellipse


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X['Longitude'], X['Latitude'], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X['Longitude'], X['Latitude'], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def main_runner():
    df_place_info = pd.read_csv('../../data/info_location.csv')

    df_long_lang = df_place_info[['Longitude', 'Latitude']]

    # plot the data
    plt.scatter(df_place_info['Longitude'], df_place_info['Latitude'])

    plt.show()

    gmm = GaussianMixture(n_components=4, random_state=44).fit(df_long_lang)
    labels = gmm.predict(df_long_lang)

    # plot the data
    plt.scatter(df_place_info['Longitude'], df_place_info['Latitude'], c=labels, s=40, cmap='viridis');

    plt.show()

    plot_gmm(gmm, df_long_lang)

    plt.show()

    """ 
    Find optimal number of componets/clusters
    
    The optimal number of clusters is the value 
    that minimizes the AIC or BIC, depending on 
    which approximation we wish to use. 
    """

    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df_long_lang)
              for n in n_components]

    plt.plot(n_components, [m.bic(df_long_lang) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(df_long_lang) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');

    plt.show()


if __name__ == "__main__":
    main_runner()
