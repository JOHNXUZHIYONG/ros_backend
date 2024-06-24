from mpmath import *
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.special import factorial
import itertools
from scipy import io as spio

# from scipy.io import loadmat
# indices = loadmat('indices_4_int.mat')
# rowidx = loadmat('rowidx_4_int.mat')

##
# This function generates uniformly distributed random variables
def uniform_RV(a, b, numSamples):
    return np.random.uniform(low=a, high=b, size=[numSamples])

##
# This function generates Rician fading random variables
def ricianFading_RV(K_factor, numSamples):
    # Generate z1 as i.i.d normal distributed RVs with unit variance and 0 mean
    z1 = np.random.normal(0, 1, numSamples)
    z1 = z1 - np.mean(z1)
    z1 = 1/(np.sqrt(np.var(z1))) * z1

    # Generate z2 as i.i.d normal distributed RVs with unit variance and 0 mean
    z2 = np.random.normal(0, 1, numSamples)
    z2 = z2 - np.mean(z2)
    z2 = 1/(np.sqrt(np.var(z2))) * z2

    # Generate X as Rician RV
    X = np.sqrt(K_factor/(1+K_factor)) + np.sqrt(1/(1+K_factor)) * (1/np.sqrt(2)) * (z1 + z2*1j)

    return np.abs(X)**2

##
# This function generates Euclidean distances as random variables following a BPP 
def euclidean_dist(ra, H, numSamples):
    dist = np.zeros(numSamples)
    # Generate (x,y) coordinates
    r = uniform_RV(0,ra,numSamples)
    theta = uniform_RV(0,2*pi,numSamples)

    for i in range(0, len(theta)):
        x = sqrt(ra*r[i])*cos(theta[i])
        y = sqrt(ra*r[i])*sin(theta[i])
        dist[i] = sqrt(x**2 + y**2 + H**2)
    return dist

##
# This function generates Nakagami-m fading random variables
def nakFading_RV(m, numSamples):
    # Generate X as Gamma distributed RV
    X = np.random.gamma(m, 1/m,numSamples)

    return X

##
# This function calculates the cumulative distribution function for Nakagami-m random variables
def nakFading_RV_CDF(m, Omega, thres, trunc_Order):
    f = lambda n: ( (-1)**n * (m/Omega)**(m+n) ) / ( gamma(m)*factorial(int(n))*(m+n) ) * thres**(m+n)
    return (nsum(f, [0,trunc_Order]))

##
# This function generates Euclidean distances as random variables following a BPP in an annular region
def euclidean_dist_annulus(min_rad, ra, H, numSamples):
    dist = np.zeros(numSamples)
    # Generate (x,y) coordinates
    v = uniform_RV(0,1,numSamples)
    theta = uniform_RV(0,2*pi,numSamples)
    
    x = np.zeros(numSamples)
    y = np.zeros(numSamples)
    dist = np.zeros(numSamples)

    for i in range(0, len(dist)):
        x[i] = sqrt( min_rad**2 + v[i]*(ra**2-min_rad**2) )*cos(theta[i])
        y[i] = sqrt( min_rad**2 + v[i]*(ra**2-min_rad**2) )*sin(theta[i])

        dist[i] = sqrt(x[i]**2 + y[i]**2 + H**2)
    return dist

##
# This function generates Euclidean distances as random variables following a PPP
def euclidean_dist_ppp(ra, H, lambda0, numSamples):
    # Association distance as the distance to the nearest BS
    assoc_dist = np.zeros(numSamples)
    assoc_dist = []
    coords_x = []
    coords_y = []

    for k in range(0,numSamples):
        numPoints = np.random.poisson(lambda0*pi*ra**2)
        if numPoints> 1:
            # numPoints = np.random.poisson(lambda0*pi*ra**2)
        
            # Generate (x,y) coordinates
            r = uniform_RV(0,ra,numPoints)
            theta = uniform_RV(0,2*pi,numPoints)
            dist = np.zeros(numPoints)

            for i in range(0, len(theta)):
                x = sqrt(ra*r[i])*cos(theta[i])
                y = sqrt(ra*r[i])*sin(theta[i])
                dist[i] = sqrt(x**2 + y**2 + H**2)
                coords_x.append(x)
                coords_y.append(y)

            assoc_dist.append(np.ndarray.min(dist))
    return assoc_dist, coords_x, coords_y

##
# This function generates ordered Euclidean distances as random variables following a BPP in an annular region
def ordered_euclidean_dist(k, min_rad, ra, H, N, numSamples):
    ordered_dist = np.zeros(numSamples)
    
    for i in range(0,numSamples):
        dist = np.zeros(N)
        for j in range(0,N):
            dist[j] = euclidean_dist_annulus(min_rad, ra, H, 1)
        ordered_dist[i] = np.sort(dist)[k-1]
    return ordered_dist

##
# This function calculates the probability density function for Euclidean distance random variables following a BPP in an annulus
def dist_PDF(min_rad, ra, w_k):
    return 2/(ra**2 - min_rad**2)*w_k

##
# This function calculates the cumulative distribution function for Euclidean distance random variables following a BPP in an annulus
def dist_CDF(min_rad, ra, thres):
    computed = ( (thres)**2 - (min_rad)**2 )/(ra**2 - min_rad**2)
    return computed

##
# This function calculates the probability density function for ordered Euclidean distance random variables following a BPP in an annulus
def ordered_dist_PDF(u, k, min_rad, ra, N):
    f = factorial(int(N))/( factorial(int(k-1)) * factorial(int(N-k)) ) \
        * dist_CDF(min_rad, ra, u)**(k-1) * (1-dist_CDF(min_rad, ra, u))**(N-k) \
        * dist_PDF(min_rad, ra, u)
    return f

##
# This function calculates the cumulative distribution function for ordered Euclidean distance random variables following a BPP in an annulus
def ordered_dist_CDF(k, min_rad, ra, N, thres):
    f = lambda w_u: ordered_dist_PDF(w_u, k, min_rad, ra, N)
    return quad(f, [min_rad,thres], method="gauss-legendre")

##
# This function calculates the effect of stochastic geometry on outage probability
def delta_func(m_x0, p, q, ra, l):
    return 2*ra**( l*(m_x0+p-q) ) / ( l*(m_x0+p-q) + 2 )

##
# This function calculates the effect of stochastic geometry on outage probability for the closest interferer in an annulus
def theta_func(q, ra, rcs, l):
    return 2 * ( ra**(2-l*q)  - rcs**(2-l*q)) / ( (ra**2 - rcs**2) * (2 - l*q) )

##
# This function calculates the CDF expansion of the Gamma distribution
def alpha_func(x0, m_x0, p, thres):
    return (-1)**p * (m_x0 * thres / x0)**(m_x0+p) / ( gamma(m_x0) * factorial(int(p)) * (m_x0+p) )

##
# This function calculates the CDF expansion of the non central Chi-sqaured distribution
def alpha_Rician_func(Omega, K_factor, p, thres):
    # return (-1)**p * exp(-K_factor) * eval_genlaguerre(int(p), int(0), float(K_factor)) / (factorial(int(1+p)) * Omega**(p+1)) \
    #     * ( (1+K_factor) * thres )**(p+1)
    return (-1)**p * exp(-K_factor) * eval_genlaguerre(int(p), int(0), float(K_factor))  / factorial(int(1+p)) \
        * ( (1+K_factor) * thres )**(p+1) \
        * Omega**(-p-1)

##
# This function calculates the normalized CDF expansion of the Gamma distribution
def norm_alpha_func(m_x0, p, thres):
    return (-1)**p * (m_x0 * thres)**(m_x0+p) / ( gamma(m_x0) * factorial(int(p)) * (m_x0+p) )

##
# This function calculates the fractional moments for Nakagami-m fading interferers
def fractionalMoments_nakFading(m_mu, mu, q):
    return (mu/m_mu)**q * gamma(m_mu + q)/gamma(m_mu)

##
# This function calculates the normalized fractional moments for Nakagami-m fading interferers
def norm_fractionalMoments_nakFading(m_Omega, m_mu, p, q):
    return binomial(m_Omega+p, q) * gamma(m_mu + q)/(gamma(m_mu) * (m_mu)**q)

##
# This function calculates the fractional moments for Rician fading interferers
def fractionalMoments_ricianFading(Omega, q, K_factor):    
    return gamma(q+1)/((1+K_factor) )**q * hyp1f1(-q, 1, -K_factor) * (Omega)**q

def Theta_func(m_Omega, epsilon, p, q):
    return (1 - epsilon**2)**q / (epsilon**2)**(m_Omega+p)

##
# This function calculates the outage probability for Nakagami-m fading, where the interferer location follows a BPP in a circular region
def outage_pwr_series_SINR_nakFading(Omega, mu, m_Omega, m_mu, thres, ra, l, M, trunc_Order):
    f = lambda p: alpha_func(Omega, m_Omega, p, thres) \
        * nsum(lambda q: \
            fractionalMoments_nakFading(m_Omega, m_mu, mu, p, q) * delta_func(m_Omega, p, q, ra, l), \
            [0,m_Omega+p])

    return (nsum(f, [0,trunc_Order]))**M

##
# This function calculates the asymptotic outage probability for Nakagami-m fading, where the interferer location follows a BPP in a circular region
def outage_pwr_series_SINR_nakFading_asymptotic(epsilon, m_Omega, m_mu, thres, ra, l, M, trunc_Order):
    f = lambda p: norm_alpha_func(m_Omega, p, thres) \
        * norm_fractionalMoments_nakFading(m_Omega, m_mu, p, m_Omega+p) \
        * delta_func(m_Omega, p, m_Omega+p, ra, l) \
        * Theta_func(m_Omega, epsilon, p, m_Omega+p)
    return (nsum(f, [0,trunc_Order]))**M

##
# This function calculates the outage probability for Nakagami-m fading, where the interferer location follows a BPP in an annular region
def outage_pwr_series_SINR_nakFading_annulus(omega, omega_icv, m_omega, m_omega_icv, thres, rcs, ra, l, M, trunc_Order):
    f = lambda p: alpha_func(omega, m_omega, p, thres) \
        * nsum(lambda q: \
            fractionalMoments_nakFading(m_omega, m_omega_icv, omega_icv, p, q) \
            * delta_func(m_omega, p, 0, ra, l) * theta_func(q, ra, rcs, l), \
            [0,m_omega+p])

    return (nsum(f, [0,trunc_Order]))**M

##
# This function calculates the asymptotic outage probability for Nakagami-m fading, where the interferer location follows a BPP in an annular region
def outage_pwr_series_SINR_nakFading_annulus_asymptotic(m_omega, m_omega_icv, thres, rcs, ra, l, M, trunc_Order):
    f = lambda p: norm_alpha_func(m_omega, p, thres) \
        * norm_fractionalMoments_nakFading(m_omega, m_omega_icv, p, m_omega+p) \
        * delta_func(m_omega, p, 0, ra, l) * theta_func(m_omega+p, ra, rcs, l)
    return (nsum(f, [0,trunc_Order]))**M

##
# This function returns multinomial indices for M_int interferers
def load_multinom_idx(M_int, tr=40, n=0):
    multinom_idx = []
    for i in range(tr+1):
        for p in partitions(i+n, M_int+1):
            multinom_idx.append(p)
    return multinom_idx

##
# This function returns combinations of elements
def partitions(n, k):
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]

def main():

    fc = 5.9e9 # fc in Hz
    T = 0.5e-3 # Feedback period in s, i.e., feedback latency
    v = 200/3.6 # Vehicle speed in m/s, input in Km/h. Tested 30km/h, 40km/h, 60km/h
    fd = v*fc/(3*10**8) # Doppler frequency
    z = 2*pi*fd*T
    z = 2.40481
    print(j0(z))
    print(nsum(lambda k: (-1/4 * z**2)**k / (factorial(int(k)) * gamma(k+1)), [0,inf]))
    z = 1.521
    print(j0(z))
    print(nsum(lambda k: (-1/4 * z**2)**k / (factorial(int(k)) * gamma(k+1)), [0,inf]))
    print(1.521*(3*10**8)/(2*pi*fc*T) * 3.6)


    z = 1.1263
    print(j0(z))
    print(1.126*(3*10**8)/(2*pi*fc*T) * 3.6)
    print(sqrt(1/2))
    e=sqrt(1/2)-1e-2
    print((1-e**2)/(e**2))
    print("")
    
    # Start of Nakagami-m Fading test
    # m = 1
    # Omega = 10**(10/10)
    # thres = 2
    
    # h_i = nakFading_RV(m,numSamples)
    # pr_h_i_sim = len([i for i in h_i if Omega*np.abs(i)<= thres])/len(h_i)
    # pr_h_i_computed = nakFading_RV_CDF(m, Omega, thres)
    # print(pr_h_i_sim)
    # print(pr_h_i_computed)

    # End of Nakagami-m Fading test

    # Start of Poisson point process test

    # ra = 2.0
    # H = 0.01
    # lambda0 = 1
    # thres = 0.5
    # assoc_dist, x, y = euclidean_dist_ppp(ra, H, lambda0, 10**3)
    # pr_assoc_dist_sim = len([i for i in assoc_dist if i<= thres])/len(assoc_dist)
    # pr_assoc_dist_computed = 1 - exp(-lambda0 * pi*thres**2)
    # print(pr_assoc_dist_sim)
    # print(pr_assoc_dist_computed)
    # plt.scatter(x,y,s=[5])
    # plt.show()

    # End of Poisson point process test

    # Start of annulus test

    ra = 2000
    H = 0
    min_rad = 500
    thres = 600
    N = 4

    dist_u = ordered_euclidean_dist(1, min_rad, ra, H, N, numSamples)
    pr_dist_sim = len([i for i in dist_u if i<= thres])/len(dist_u)
    # pr_dist_sim = len([np.min(i) for i in dist_u if np.min(i)<= thres])/len(dist_u)
    pr_dist_computed = ordered_dist_CDF(1, min_rad, ra, N, thres)
    print(pr_dist_sim)
    print(pr_dist_computed)

    # dist, x, y = euclidean_dist_annulus(min_rad, ra, H, 10**4)
    # pr_dist_sim = len([i for i in dist if i<= thres])/len(dist)
    # pr_dist_computed = ( thres**2 - min_rad**2 )/(ra**2-min_rad**2)
    # # pr_dist_computed = ( (thres)**2 )/(ra**2)
    # print(pr_dist_sim)
    # print(pr_dist_computed)
    # plt.scatter(x,y,s=[5])
    # plt.show()

    # End of annulus test

# Runs main()
if __name__ == "__main__":
    main()