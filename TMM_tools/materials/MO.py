from TMM_tools.materials.base_material import *
from TMM_tools.core.constants import *
import numpy as np

e_inf = 12.37
def Exx(w,w_p,w_c,T):
    exx=e_inf-(w_p**2*(w+1j*T))/(w*((w+1j*T)**2-w_c**2))
    return exx
def Exz(w,w_p,w_c,T):
    exz=-1j*(w_p**2*w_c)/(w*((w+1j*T)**2-w_c**2))
    return exz
def Eyy(w,w_p,T):
    eyy=e_inf-w_p**2/(w*(w+1j*T))
    return eyy
def Wp(ne,m_):
    wp=np.sqrt(ne*e**2/m_/epsilon_0)
    return wp
def Wc(B,m_):
    wc=e*B/m_
    return wc
def MO_transfer_Matrix(delta,eta,exx,exz,theta):
    theta = theta * pi / 180
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    tan_theta = np.tan(theta)
    exz_exx=exz/exx
    M=np.array([[  cos_delta+exz_exx*tan_theta*sin_delta  ,  -1j*sin_delta*(1+(exz_exx*tan_theta)**2)/eta  ],
                [  -1j*eta*sin_delta                        ,  cos_delta-exz_exx*tan_theta*sin_delta        ]])
    return M
class MO_Material(Basic_Material):
    def __init__(self,T,m_,ne,B,mu=1,mode="TM"):
        super().__init__()
        self.T=T
        self.m_=m_
        self.ne=ne
        self.B=B
        self.mu=mu
        self._wp=Wp(self.ne,self.m_)
        self._wc=Wc(self.B,self.m_)
        self.mode=mode
    def exx(self,f):
        w=2*pi*f
        exx=Exx(w,self._wp,self._wc,self.T)
        return exx
    def exz(self,f):
        w=2*pi*f
        exz=Exz(w,self._wp,self._wc,self.T)
        return exz
    def eyy(self,f):
        w=2*pi*f
        eyy=Eyy(w,self._wp,self.T)
        return eyy
    def get_n(self,f):
        if self.mode=="TM":
            exx=self.exx(f)
            exz=self.exz(f)/1j
            n=np.sqrt((exx**2-exz**2)/exx)
        else:
            eyy=self.eyy(f)
            n=np.sqrt(eyy)
        return n
    def get_epsilon(self,f):
        n=self.get_n(f)
        epsilon=n**2
        return epsilon
    def get_mu(self,f):
        return self.mu
if __name__=='__main__':
    ...