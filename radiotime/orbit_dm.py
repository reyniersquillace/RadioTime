import astropy.units as u 
import astropy.constants as const
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad

class orbit:

    def __init__(self, T, M, m, e, i, w):

        self.T = T
        self.M = M
        self.m = m
        self.e = e
        self.a = self.semimajor()
        self.b = self.semiminor()
        self.p = self.get_p()
        self.i = i
        self.w = w
        self.c = np.sqrt(self.a**2 - self.b**2)
        self.closest_approach = self.a - self.c


    def semimajor(self):

        return ((self.T**2 * const.G * (self.M + self.m) / 4 / np.pi**2)**(1/3)).to(u.AU)

    def semiminor(self):

        return self.a * np.sqrt(1 - self.e)

    def get_p(self):

        return self.b**2 / np.sqrt(self.a**2 - self.b**2)

    def A_from_t(self, t):

        return (np.pi * self.a * self.b * t / self.T).to(u.AU**2)

    def r_from_t(self, t):

        A = self.A_from_t(t)

        return self.p / (1 - self.e * np.cos(2 * A * u.rad/ self.a / self.b))

    def solve_kepler_equation(self, M, tol=1e-9, max_iter=100):
        """
        From ChatGPT.
        
        Solve M = E - e*sin(E) for E using Newton's method.
        M, E in radians, e dimensionless.
        """
        # Normalize M to be between -pi and +pi if desired (not strictly necessary)
        M = np.mod(M, 2*np.pi)
        if M > np.pi:
            M -= 2*np.pi
        
        # Initial guess for E (a simple guess: E = M)
        E = M if self.e < 0.8 else np.pi  # or some other heuristic if e is large
        
        for _ in range(max_iter):
            f = E - self.e*np.sin(E) - M
            fprime = 1 - self.e*np.cos(E)
            E_new = E - f/fprime
            if abs(E_new - E) < tol:
                return E_new
            E = E_new

    def nu_from_t(self, t):

        mean_anomaly = (2 * np.pi * t / self.T).value
        eccentric_anomaly = self.solve_kepler_equation(mean_anomaly)
        nu = 2 * np.arctan(np.sqrt((1 + self.e)/(1 - self.e)) * np.tan(eccentric_anomaly / 2))

        return nu

    def alpha_from_nu(self, nu):

        return np.arccos(np.sin(self.i) * np.sin(self.w + nu))
                
    def DM_mock(self, t, K):

        nu = self.nu_from_t(t)
        alpha = self.alpha_from_nu(nu, self.w).value
        r = self.r_from_t(t)

        return (K*alpha / r / np.sin(alpha)).to(u.pc * u.cm**(-3))
        