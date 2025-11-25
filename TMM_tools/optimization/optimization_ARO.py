import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
import os

__code_author__='Chengyi Li'

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

#相关日志记录
def _putout(text,file=None):
    if file is None:
        ...
    elif file==sys.stdout:
        print(text,file=file)
    else:
        print(text,file=sys.stdout)
        with open(file,"a+") as f:
            print(text,file=f)
#获得当前时间戳
def _get_time(forsave=False):
    if not forsave:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    else:
        return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
#转换时间格式
def _change_time(time):
    Secons=Minius=60
    Hors=24
    dys=time//(Hors*Minius*Secons)
    time=time%(Hors*Minius*Secons)
    hors=time//(Minius*Secons)
    time=time%(Minius*Secons)
    minus=time//(Secons)
    secons=time%(Secons)
    return f"{dys:.0f} 天 {hors:.0f} 时 {minus:.0f} 分 {secons:.3f} 秒"
#随机隐藏候选位置策略
def _random_hidding(i, t, dim, Population, MaxIteration):
    H=(MaxIteration-t+1)/MaxIteration*np.random.rand()
    fr=np.zeros(dim)
    id=np.random.randint(dim)
    fr[id]=1
    b=Population[i]+H*fr*Population[i]
    Pos=Population[i]+_R(t,MaxIteration,dim)*(np.random.rand()*b-Population[i])
    return Pos
#奔跑维度向量确定
def _R(t,T,dim):
    e = np.exp(1)
    L = ((e - np.exp(((t - 1) / T) ** 2)) *
         np.sin(2 * np.pi * np.random.rand()))
    c = np.zeros(shape=dim)
    num = np.random.randint(1, dim + 1)
    idx = random.sample(range(dim), num)
    c[idx] = 1
    return c*L
#迂回觅食侯选位置策略
def _detour_seeking(i, t, dim, Population, MaxIteration, PopulationSize):
    j=np.random.randint(PopulationSize)
    while j==i:
        j=np.random.randint(PopulationSize)
    Pos=Population[j]+_R(t,MaxIteration,dim)*(Population[i]-Population[j])+round(0.5*(0.05+np.random.rand()))*np.random.normal(1)
    return Pos
#边界调整函数
def _regulate_position(lb, ub, Pos, regulate_method="cut"):
    '''"cut","rebound"'''
    Over_ub=Pos>ub
    Over_lb=Pos<lb
    if regulate_method=="cut":
        Pos[Over_ub]=ub[Over_ub]
        Pos[Over_lb]=lb[Over_lb]
    elif regulate_method=="rebound":
        Pos[Over_ub]=ub[Over_ub]-np.mod(Pos[Over_ub]-ub[Over_ub],ub[Over_ub]-lb[Over_ub])
        Pos[Over_lb]=lb[Over_lb]+np.mod(lb[Over_lb]-Pos[Over_lb],ub[Over_lb]-lb[Over_lb])
    else:
        raise ValueError("regulate_method must be 'cut' or 'rebound'")
    return Pos
#兔子能量因子计算
def _A(t,T):
    r=np.random.rand()
    a=4*(1-t/T)*np.log(1/r)
    return a
#初始化函数
def _init(lb,ub,dim):
    Pos=np.zeros(shape=dim)
    r=np.random.random(dim)
    Pos=lb+r*(ub-lb)
    return Pos

def load_Iteration_Figure(checkpoint_file):
    data=np.load(checkpoint_file)
    t_Max=data["t_current"]
    best_Epoch_fitness=data["best_Epoch_fitness"]
    x=range(t_Max)
    plt.figure("Iteration-checkpoint",figsize=(8,6))
    plt.plot(x,best_Epoch_fitness[:t_Max],label="checkpoint-iteration")
    plt.xlabel("iteration number")
    plt.ylabel("fitness value")
    plt.title("Iteration of checkpoint")
    plt.legend()
    plt.show()

def ARO(Func,lb,ub,MaxIteration,PopulationSize,
          fitness_outformat='<15.5E',Round_func=None,extra_data=None,regulate_method='cut',
          log_file=sys.stdout,logtime=True,show_iteration=False,print_pocession=True,
          checkpoint_interval=0,checkpoint_file=None,checkpoints_dirheater=None):
    """
    人工兔优化算法
    参数表：
    Func：适应度函数
    lb：搜索下界
    ub：搜索上界
    MaxIteration：最大迭代次数
    PopulationSize：种群数，种群规模
    fitness_outformat：适应度日志打印格式
    Round_func：可自定义的精度约束函数
    extra_data：额外数据传输，用于适应度计算时，适应度函数的某些参数随个体而改变，设计用于完成拟合与求解方程任务
    regulate_method：优化搜索时边界模式，可选剪切"cut"，反弹"rebound"
    log_file：日志记录，默认为控制台输出，为None时隐藏所有信息，为文件路径时控制台与文件同时记录
    logtime：迭代时间记录，包括每轮迭代时间与总时间
    show_iteration：迭代收敛曲线展示，默认关闭，支持展示
    print_pocession：种群个体计算适应度打印，默认展示，关闭时只记录每轮迭代的最优与耗时信息
    checkpoint_interval：存档点间隔，默认为0，表示不进行存档；为大于零的整数时，在迭代对应轮数时保存迭代现场
    checkpoint_file：重加载时的存档点文件，传入对应文件路径时从文件中加载迭代数据继续迭代
    返回值：
    best_fitness：全局最优适应度
    best_Pos：全局最优位置
    best_Epoch_fitness：迭代历史最优适应度
    best_Epoch_Pos：迭代历史最优位置
    -------------------------------------------------------
    Artificial Rabbit Optimization Algorithm
    Parameter Table:
    Func: Fitness Function
    lb: Search Lower Bound
    ub: Search Upper Bound
    MaxIteration: Maximum Number of Iterations
    PopulationSize: Number of Populations, Population Size
    fitness_outformat: Fitness Log Printing Format
    Round_func: Customizable Precision Constraint Function
    extra_data: Extra Data Transmission, Used for Fitness Calculation,
                Some parameters of the fitness function change with the individual,
                designed to complete fitting and solving equation tasks.
    regulate_method: Optimize the boundary mode when searching,
                     optional cut "cut", rebound "rebound".
    log_file: logging, default to console output,
              hide all information when None.
    logtime: log when the console and file are recorded at the same time as the file path,
             including each round of iteration time and total time.
    show_ iteration: Iterative convergence curve display,
                     turned off by default, and supports display.
    print_pocession: Population individual fitness printing, default display,
                     only record the optimal and time-consuming information
                     of each round of iteration when turned off.
    checkpoint_interval: Archive point interval, default to 0, indicating no archiving;
                        If it is an integer greater than zero, save the iteration site
                        when iterating the corresponding number of rounds.
    checkpoint_file: Save the file when reloading, and continue iterating from the file
                     when the corresponding file path is passed in .
    Return value:
    best_fitness: Global optimal fitness
    best_Pos: Global optimal position
    best_Epoch_fitness: Optimal fitness in iteration history
    best_Epoch_Pos: Optimal position in iteration history
    """

    lb,ub=np.array([lb,ub])
    dim=len(lb)

    #异常输入
    if not callable(Func):
        raise ValueError("Func must be callable")
    if (Round_func is not None) and (not callable(Round_func)):
        raise ValueError("Round_func must be callable")
    if not len(lb)==len(ub):
        raise ValueError("lb and ub must have same length")
    if MaxIteration<=0:
        raise ValueError("MaxIteration must be positive")
    if PopulationSize<2:
        raise ValueError("PopulationSize must be greater than 1")
    if np.any(lb>ub):
        raise ValueError("lb and ub must be strictly increasing")
    if np.any(np.abs(ub - lb) < 1e-30) and regulate_method=="rebound":
        raise ValueError("ub and lb must be different While regulate_method=rebound")

    #配置信息
    _putout(f"-----------开始运行{ARO.__name__}-----------\n"
            f"运行配置如下:\n\n"
            f"{'Function':^26} | {Func.__name__}\n"
            f"{'Low boundary':^26} | {lb}\n"
            f"{'Up boundary':^26} | {ub}\n"
            f"{'Dimension':^26} | {dim}\n"
            f"{'Max iteration':^26} | {MaxIteration}\n"
            f"{'Population size':^26} | {PopulationSize}\n"
            f"{'Fitness output format':^26} | {fitness_outformat}\n"
            f"{'Round func':^26} | {Round_func.__name__ if Round_func is not None else 'None'}\n"
            f"{'Extra data':^26} | {'None' if extra_data is None else 'True'}\n"
            f"{'Regulate method':^26} | {regulate_method}\n"
            f"{'Log file path':^26} | {'sys.stdout' if log_file==sys.stdout else log_file}\n"
            f"{'Log time':^26} | {logtime}\n"
            f"{'Show iteration':^26} | {show_iteration}\n"
            f"{'print_pocession':^26} | {print_pocession}\n"
            ,file=log_file)
    start_time = time.time()
    if checkpoint_file is not None:
        #重加载
        load_data=np.load(checkpoint_file)
        t_start=load_data["t_current"]+1
        best_Pos,best_fitness, counter ,best_count=load_data["best_Pos"], load_data["best_fitness"], int(load_data["counter"]),int(load_data["best_count"])
        Population,Fitness=load_data["Population"],load_data["Fitness"]
        best_Epoch_fitness,best_Epoch_Pos=load_data["best_Epoch_fitness"],load_data["best_Epoch_Pos"]
        _putout(f"{' | '+_get_time()+' 从第'+str(t_start)+'轮重加载优化算法 | ':-^120}",file=log_file)
        _putout(f"重加载配置如下:\n\n"
                f"{'t of start':^20} | {t_start}\n"
                f"{'Best fitness before':^20} | {best_fitness}\n"
                f"{'Best position before':^20} | {best_Pos}\n"
                f"{'Best run id':^20} | {best_count}\n",file=log_file)
    else:
        #初始化
        Population,Fitness=np.zeros(shape=(PopulationSize,dim)),np.zeros(shape=PopulationSize)
        _putout(f"{' | '+_get_time() + '  开始种群初始化 | ' if logtime else '':-^120}", file=log_file)
        for i in range(PopulationSize):
            Population[i]=_init(lb,ub,dim)
            Population[i] = Round_func(Population[i]) if Round_func is not None else Population[i]
            Fitness[i]=Func(Population[i],extra_data=extra_data) if extra_data is not None else Func(Population[i])
            _putout("初始化种群个体{0:^4}适应度: {1:{2}} 对应x: {3}"
                    .format(i+1,Fitness[i],fitness_outformat,np.array2string(Population[i],separator=','))
                    ,file=log_file) if print_pocession else ...
        #数据记录
        counter=0
        t_start=1
        best_count=0
        best_fitness,best_Pos=np.min(Fitness),Population[Fitness.argmin()]
        best_Epoch_fitness,best_Epoch_Pos=np.zeros(shape=MaxIteration+1),np.zeros(shape=(MaxIteration+1,dim))
        best_Epoch_fitness[0],best_Epoch_Pos[0]=best_fitness,best_Pos
        _putout(f"{' | '+_get_time()+'  完成种群初始化 | '+f'总耗时:{_change_time(time.time() - start_time)} | ' if logtime else '':-^120}",file=log_file)
        _putout("初始化最优适应度: {0:{1}} 运行号:{2:<10} 对应x: {3}"
                .format(best_fitness,fitness_outformat,best_count,np.array2string(best_Pos,separator=','))
                ,file=log_file)
    start_epoch_time=time.time()
    #主迭代过程
    for t in range(t_start,MaxIteration+1):
        _putout(f"{' | '+_get_time() + f' 开始第{t}轮迭代 | ' + f'上轮耗时:{_change_time(time.time() - start_epoch_time)} | '+ f'总耗时:{_change_time(time.time() - start_time)} |' if logtime else '':-^120}"
            ,file=log_file)
        start_epoch_time=time.time()
        #遍历整个种群
        for i in range(PopulationSize):
            #兔子能量
            A=_A(t,MaxIteration)
            if A>1:#迂回觅食
                obj_position=_detour_seeking(i, t, dim, Population, MaxIteration, PopulationSize)
            else:#随机隐藏
                obj_position=_random_hidding(i, t, dim, Population, MaxIteration)
            obj_position=Round_func(obj_position) if Round_func is not None else obj_position       #自定义精度
            obj_position=_regulate_position(lb, ub, obj_position,regulate_method=regulate_method)   #边界调整
            obj_fitness=Func(obj_position,extra_data=extra_data) if extra_data is not None else Func(obj_position)  #评估适应度
            counter+=1
            _putout("第{0}次适应度F: {1:{2}} 对应x: {3}"
                    .format(counter, obj_fitness, fitness_outformat, np.array2string(Population[i],separator=','))
                    , file=log_file) if print_pocession else ...
            if obj_fitness<Fitness[i]:
                #个体位置更新
                Fitness[i],Population[i]=obj_fitness,obj_position
            if obj_fitness<best_fitness:
                #最优更新
                best_count,best_fitness,best_Pos = counter,obj_fitness,obj_position
        _putout("最优适应度F: {0:{1}} 运行号:{2:<10} 对应x: {3}"
                .format(best_fitness, fitness_outformat, best_count, np.array2string(best_Pos,separator=','))
                , file=log_file)
        best_Epoch_fitness[t],best_Epoch_Pos[t]=best_fitness,best_Pos       #历史最优记录
        if checkpoint_interval>0 and t%checkpoint_interval==0:
            #存档点记录
            checkpoints_dirheater=os.path.splitext(os.path.basename(sys.argv[0]))[0] if checkpoints_dirheater is None else checkpoints_dirheater
            os.makedirs(f"checkpoints-{checkpoints_dirheater}",exist_ok=True)
            save_data_dict={"t_current":t,"Population":Population,
                            "Fitness":Fitness,"best_fitness":best_fitness,
                            "best_Pos":best_Pos,"best_Epoch_fitness":best_Epoch_fitness,
                            "best_Epoch_Pos":best_Epoch_Pos,"counter":counter,"best_count":best_count}
            save_file=f"checkpoints-{checkpoints_dirheater}/checkpoint-{_get_time(forsave=True)}.npz"
            np.savez(save_file,**save_data_dict)
            _putout(f"{f' | checkpoint saved at {save_file} | ':*^110}",file=log_file)
    _putout(f"{' | '+_get_time() + f' 完成所有迭代 | ' + f'末轮耗时:{_change_time(time.time() - start_epoch_time)} | ' + f'总耗时:{_change_time(time.time() - start_time)} | ' if logtime else '':-^120}",file=log_file)
    _putout( f"运行结果如下:\n"
            f"{'Best fitness':^26} | {best_fitness}\n"
            f"{'Best position':^26} | {np.array2string(best_Pos,separator=',')}\n"
            f"{'Run id':^26} | {best_count}\n"
            ,file=log_file)

    #绘制收敛曲线
    if show_iteration:
        plt.figure("Iteration Fitness Figure",figsize=(8,6),dpi=100)
        plt.plot(range(MaxIteration+1),best_Epoch_fitness,label="fitness curve",ls="-")
        plt.legend(loc="best")
        plt.title("Iteration Fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.show()
    return best_fitness,best_Pos,best_Epoch_fitness,best_Epoch_Pos

if __name__ == '__main__':

    def Func(x):
        F=0
        for i in range(len(x)):
           F+=x[i]**2
        return F
    def My_round_func(x):
        for i in range(len(x)):
            if i <10:
                x[i]=round(x[i],1)
            else:
                x[i]=round(x[i],2)
        return x
    lb=[-30]*100
    ub=[30]*100
    MaxIt=1000
    L=200
    ARO(Func,lb,ub,MaxIt,L,log_file=sys.stdout,checkpoint_interval=600,
          checkpoint_file=None,print_pocession=False,Round_func=None,
          show_iteration=False,checkpoints_dirheater=None)
    ...
