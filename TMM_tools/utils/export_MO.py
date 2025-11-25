import sys

from TMM_tools import dispersion_fit
from TMM_tools.materials.MO import *

def _print_com(params,f_mean,f_std,file):
    print(f"({params[0]:.5e})*((f-{f_mean:.5e})/{f_std:.5e})^3+"
          f"({params[1]:.5e})*((f-{f_mean:.5e})/{f_std:.5e})^2+"
          f"({params[2]:.5e})*((f-{f_mean:.5e})/{f_std:.5e})^1+"
          f"({params[3]:.5e})",file=file)
def export_MO(flis,T,m_,ne,B,print_path=None):
    InAs=MO_Material(T,m_,ne,B)
    InAs_Exx=np.zeros(shape=len(flis),dtype=np.complex128)
    InAs_Exz=InAs_Exx.copy()
    InAs_Eyy=InAs_Exz.copy()
    for i,f in enumerate(flis):
        InAs_Exx[i]=InAs.exx(f)
        InAs_Exz[i] = InAs.exz(f)
        InAs_Eyy[i] = InAs.eyy(f)
    f_mean,f_std=np.mean(flis),np.std(flis)
    params_exx_r,params_exx_i=dispersion_fit(flis,InAs_Exx)
    params_exz_r,params_exz_i=dispersion_fit(flis,InAs_Exz)
    params_eyy_r, params_eyy_i = dispersion_fit(flis, InAs_Eyy)
    if print_path is not None:
        with open(print_path,"w") as f:
            print(f"T: {T:4e}  m_eff: {m_/me:.4e}  ne: {ne:.4e}  B: {B:.4e}",file=f)
            print("Exx:",file=f)
            _print_com(params_exx_r,f_mean,f_std,f)
            _print_com(params_exx_i,f_mean,f_std,f)
            print("Eyy:",file=f)
            _print_com(params_eyy_r,f_mean,f_std,f)
            _print_com(params_eyy_i,f_mean,f_std,f)
            print("Ezz:",file=f)
            _print_com(params_exx_r,f_mean,f_std,f)
            _print_com(params_exx_i,f_mean,f_std,f)
            print("Exz:",file=f)
            _print_com(params_exz_r,f_mean,f_std,f)
            _print_com(params_exz_i,f_mean,f_std,f)
            print("Ezx:",file=f)
            _print_com(-np.array(params_exz_r),f_mean,f_std,f)
            _print_com(-np.array(params_exz_i),f_mean,f_std,f)
    else:
        f=sys.stdout
        print(f"T: {T:4e}  m_eff: {m_ / me:.4e}  ne: {ne:.4e}  B: {B:.4e}", file=f)
        print("Exx:", file=f)
        _print_com(params_exx_r, f_mean, f_std, f)
        _print_com(params_exx_i, f_mean, f_std, f)
        print("Eyy:", file=f)
        _print_com(params_eyy_r, f_mean, f_std, f)
        _print_com(params_eyy_i, f_mean, f_std, f)
        print("Ezz:", file=f)
        _print_com(params_exx_r, f_mean, f_std, f)
        _print_com(params_exx_i, f_mean, f_std, f)
        print("Exz:", file=f)
        _print_com(params_exz_r, f_mean, f_std, f)
        _print_com(params_exz_i, f_mean, f_std, f)
        print("Ezx:", file=f)
        _print_com(-np.array(params_exz_r), f_mean, f_std, f)
        _print_com(-np.array(params_exz_i), f_mean, f_std, f)
if __name__ == '__main__':
    ...
