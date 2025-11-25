from TMM_tools.structures.Multi_Struct import *
from TMM_tools.core.constants import unitdir

class Structure_responser(Multilayer_Strcture):
    def __init__(self,layers_lis):
        super().__init__(layers_lis=layers_lis)
    def frequency_scan(self,f_lis,theta,mode="TE",upper_background_n=1,lower_background_n=1,only_T=False,only_R=False):
        Tlis=np.zeros(shape=len(f_lis))
        Rlis=Tlis.copy()
        for i, f in enumerate(f_lis):
            Tlis[i],Rlis[i],_= self.get_TRA(f, theta, mode, upper_background_n, lower_background_n)
        Alis = 1 - Tlis - Rlis
        if only_T:
            return Tlis
        elif only_R:
            return Rlis
        else:
            return Tlis,Rlis,Alis
    def angle_scan(self,f,theta_lis,mode="TE",upper_background_n=1,lower_background_n=1,only_T=False,only_R=False):
        Tlis=np.zeros(shape=len(theta_lis))
        Rlis=Tlis.copy()
        for i, theta in enumerate(theta_lis):
            Tlis[i],Rlis[i],_= self.get_TRA(f, theta, mode, upper_background_n, lower_background_n)
        Alis = 1 - Tlis - Rlis
        if only_T:
            return Tlis
        elif only_R:
            return Rlis
        else:
            return Tlis,Rlis,Alis
    def list_all(self,unit='nm',accuracy=2):
        print(f"{'name':^16}|{'thickness/'+f'({unit})':^24}|{'materialclass':^24}")
        print("-" * (16 + 24 + 24 + 2))
        for i, layer in enumerate(self._layers):
            name=layer.material.__class__.__name__
            print("{0:^16}|{1:^24.{2}f}|{3:^24}".
                  format(layer.name,layer.thickness /unitdir[unit],accuracy,
                         'Normal:  '+str(round(layer.material.get_n(0),accuracy)) if name=='Normal_Material' else name))
if __name__ == '__main__':
    ...