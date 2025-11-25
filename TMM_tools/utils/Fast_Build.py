from TMM_tools.structures.Single_layer import *

def Fast_found_Multi_Normal_Structure(Numbers, Refractives=None,
                                      Thicknesses=None, Names=None,
                                      unit="m"):
    if (Refractives is not None) and (len(Refractives)!=Numbers):
        raise ValueError("Refractives must have same length with number of layers")
    if (Thicknesses is not None) and (len(Thicknesses)!=Numbers):
        raise ValueError("Thicknesses must have same length with number of layers")
    if (Names is not None) and (len(Names)!=Numbers):
        raise ValueError("Names must have same length with number of layers")
    structlis=[]
    for i in range(Numbers):
        mat=Normal_Material(refractiveindex=
                            Refractives[i] if Refractives is not None else 1)
        layer=Single_Layer(name=Names[i] if Names is not None else f"default-{i + 1}",
                           material=mat,
                           thickness=Thicknesses[i] if Thicknesses is not None else 0,
                           unit=unit)
        structlis.append(layer)
    return structlis

if __name__=='__main__':
    import numpy as np
    Str=Fast_found_Multi_Normal_Structure(
        2,Thicknesses=[1,2],unit="pm")
    import TMM_tools.structures.Responser as res
    aaa=res.Structure_responser(Str)
    aaa.list_all(unit="nm",accuracy=3)