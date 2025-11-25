import numpy as np
from TMM_tools.materials.base_material import *
from scipy.optimize import curve_fit

def fit_func(x, a, b, c, d):
    f =a*x**3+b*x**2+c*x+d
    return f

def dispersion_fit(flis,nlis):
    nr,ni=np.real(nlis),np.imag(nlis)
    f_mean,f_std=np.mean(flis),np.std(flis)
    f_one=(flis-f_mean)/f_std
    params_nr,nrcov=curve_fit(fit_func, f_one, nr, maxfev=10000)
    params_ni,nicov=curve_fit(fit_func, f_one, ni, maxfev=10000)
    nr_expect=fit_func(f_one, *params_nr)
    R2_nr,SSE_nr=R_square(nr,nr_expect)
    if np.sum(np.abs(ni))<1e-10:
        print(f"{'R^2':^16}|{'SSE [real/imag]':^24}")
        print("-"*(16+25))
        print(f"{f'{R2_nr:.4f}/nan':^16}|{f'{SSE_nr:.4f}/nan':^24}")
    else:
        ni_expect=fit_func(f_one, *params_ni)
        R2_ni,SSE_ni=R_square(ni,ni_expect)
        print(f"{'R^2':^16}|{'SSE [real/imag]':^24}")
        print("-"*(16+25))
        print(f"{f'{R2_nr:.4f}/{R2_ni:.4f}':^16}|{f'{SSE_nr:.4f}/{SSE_ni:.4f}':^24}")
    return params_nr,params_ni

def R_square(y_origin,y_expect):
    y_ave=np.mean(y_origin)
    SST=np.sum((y_origin-y_ave)**2)
    SSR=np.sum((y_expect-y_ave)**2)
    R2=SSR/SST
    SSE=np.sum((y_expect-y_origin)**2)
    return R2,SSE

class Dispersion_Material(Basic_Material):
    def __init__(self,flis,refractiveindex_lis,mu=1,is_fit=False):
        super().__init__()
        self.is_fit=is_fit
        if is_fit:
            self.f_mean, self.f_std=np.mean(flis),np.std(flis)
            self.params_nr,self.params_ni=dispersion_fit(flis,refractiveindex_lis)
        else:
            self.flis=np.array(flis)
            self.nlis=np.array(refractiveindex_lis)
        self._mu=mu
    def get_n(self,f):
        if self.is_fit:
            f_one=(f-self.f_mean)/self.f_std
            nr=fit_func(f_one, *self.params_nr)
            ni=fit_func(f_one, *self.params_ni)
            n=nr+1j*ni
        else:
            n=self.nlis[np.abs(self.flis-f)<1]
        return n
    def get_epsilon(self,f):
        n=self.get_n(f)
        epsilon=n**2
        return epsilon
    def get_mu(self,f):
        return self._mu
if __name__=='__main__':
    ...
