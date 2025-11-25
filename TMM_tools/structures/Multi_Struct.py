from TMM_tools.core.basic import *
class Multilayer_Strcture():
    def __init__(self, layers_lis):
        self._layers=layers_lis
    def get_Transfer_Matrix(self,f,theta,mode="TE"):
        M=np.eye(2)
        for i,layer in enumerate(self._layers):
            M= M @ layer.get_transfer_matrix(f, theta, mode=mode)
        return M
    def get_TR_coefficient(self,f,theta,mode="TE",upper_background_n=1,lower_background_n=1):
        M=self.get_Transfer_Matrix(f,theta,mode=mode)
        t,r=TR_coefficient(M, theta,
                           mode=mode, upper_background_n=upper_background_n,
                           lower_background_n=lower_background_n,coefficient=True)
        return t,r

    def get_TRA(self,f,theta,mode="TE",upper_background_n=1,lower_background_n=1):
        M=self.get_Transfer_Matrix(f,theta,mode=mode)
        T,R,A=TR_coefficient(M, theta,
                             mode=mode, upper_background_n=upper_background_n,
                             lower_background_n=lower_background_n,coefficient=False)
        return T,R,A

if __name__=="__main__":
    ...