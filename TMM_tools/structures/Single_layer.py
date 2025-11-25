from TMM_tools.structures.base_layer import *
from TMM_tools.core.basic import *
from TMM_tools.core.constants import unitdir
from TMM_tools.materials.Normal import *
from TMM_tools.materials.Dispersion import *
from TMM_tools.materials.MO import *

class Single_Layer(Basic_Layer):
    def __init__(self,name="default",material=Normal_Material(),thickness=0,unit="m"):
        super().__init__(thickness=thickness*unitdir[unit])
        self.name=name
        self._material=material
    def get_transfer_matrix(self, f, theta, mode="TE"):
        '''需要注意传输矩阵中的theta为介质内的theta，不要混淆入射角与介质角'''
        n_material=self._material.get_n(f)
        theta = Theta_Reverse(1,n_material, theta)
        delta = Delta(f,n_material,self._thickness,theta)
        eta = Eta(n_material,theta,mode=mode)
        if (mode=="TM") and isinstance(self._material,MO_Material):
            exx=self._material.exx(f)
            exz=self._material.exz(f)/1j
            M=MO_transfer_Matrix(delta,eta,exx,exz,theta)
        else:
            M=Normal_material_matrix(delta,eta)
        return M
    @property
    def thickness(self):
        return self._thickness
    @thickness.setter
    def thickness(self,thickness):
        self._thickness=thickness
    @property
    def material(self):
        return self._material
    @material.setter
    def material(self,material):
        if issubclass(type(material),Basic_Material):
            self._material=material
        else:
            raise TypeError("Type Error with not Material Class")
if __name__=='__main__':
    ...