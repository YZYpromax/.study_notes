# CUDA执行模型

GPU架构（Fermi架构）SM

![1755245094017](image/CUDA执行模型学习/1755245094017.png)

![1755275030322](image/CUDA执行模型学习/1755275030322.png)

一个block只能被分配在一个SM上，但是一个SM上可以分配多个block

SIMD SIMT

CUDA采用SIMT架构管理执行线程，每个线程都有自己状态寄存器、指令地址计数器以及执行路径，使得编程模型支持每个线程的唯一编号，确保了每个线程之间的独立性

线程块内同步（需要显示调用__syncthreads())  块间不同步

divergence(分支发散)

```
__global__ void divergence(int* data){
int tid = threadIdx.x;
if(tid%2==0){
	data[tid]=1;
}else{
	data[tid]=2;
}
	__syncthreads();
}
```

SM上线程束之间的状态转换不需要开销（GPU的调度单位是wrap  一般wrap有32个线程  对应硬件的32 core)：

对比CPU状态转换  寄存器等提前分配好，不需要保存当前寄存器状态等。

wrap能够通过wrap快速切换 隐藏延迟

Giga thread  千兆线程

当一个线程块被指定给一个SM时，线程块内的所有线程被分成多个线程束，两个线程束在warp schedule的调度下在instruction Cache存储相应的指令后在SM上执行，多个线程束在这两个线程调度器的调度下 在SM上交替执行。

serial  concurrent

性能优化从程序的空间/时间复杂度、特殊指令的使用、函数调用的频率 和持续时间出发

## 线程束的硬件与逻辑形式

### 组织形式以及线程分支

三维坐标对应的线性转换：我们将三维空间视为分层结构（z 层），每层包含行（y 行），每行包含元素（x 位置）

x增长到最大 则y+1  即增加一行  直到 y增长到最大 即x.y确定的这一层 已满  则z+1
先小块  然后满行  满层  逐步填充。

则计算公式idx= threadidx.x+threadIdx*blockDim.x+threadidx.z *blockDim.x *blockDim.y

stall  execution (执行停滞)

如果将控制流的分化放在同一个线程束中，则会导致同一个线程束内有过多不符合当前判断条件的线程停滞，影响效率，为了防止线程分化带来的效率的严重下降，则可以通过分类方式将不同执行条件的线程 分别放在两个线程束中，修改上述分支发散的代码：

```
if(tid/wrapSize==0){
	data[tid]=1;
}else
	data[tid]=2;
```

当线程为64时，if else中的线程将能够在32线程上 分别执行

`((tid/warpSize)%2==0)//扩展至n个线程的场景`

### 资源分配

SM上线程束分为激活 未激活

激活：选定、符合条件、阻塞

未激活：只被分配 未上SM

寄存器 共享内存：线程越多 线程平均的资源越少

### 延迟隐藏

硬件利用率最大化 每时每刻  warp schedule都有可用的线程束可供调度

little法则
最小化延迟所需线程数=延迟*吞吐量

吞吐量 ：指实际操作过程中每分钟处理多少个指令

## 并行性表现

三元条件运算符： 条件？表达式1：表达式2  条件为真 返回式1 条件为假 返回式2

int dimx = argc>2? atoi(argv[1]):32

根据命令行输入 初始化 dimx argc是命令行参数个数 argv存储命令行参数 argc一般为运行的文件名

```
./example  32 32
```

a to i  -> str to l      argc =   `strtol(str, NULL, 10)  如果要转换为int型：需注意`

```
#include<limits.h>
if(argc<INT_MIN||ARGC>INT_MAX)//输出检查
```

不同大小块的代码执行效率与机器本身有关

### 占有率计算与区分

1.cuda下的tools中自带的xlsx CUDA GPU occupancy calculator 是理论计算的占有率

```
2.ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./simple_sum_matrix
```

该命令是实际的占有率 actived_occupancy活跃线程束比例，但是该比率高 不一定执行速度就快。

![1755436666432](image/CUDA执行模型学习/1755436666432.png)

### 内存利用率

吞吐量：nvprof --metrics gld_throughput ./simple_sum_matrix

    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second

全局加载效率：nvprof --metrics gld_efficiency ./simple_sum_matrix
	smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct

常用指标对照

| achieved_occupancy | sm__warps_active.avg.pct_of_peak_sustained_active             |
| ------------------ | ------------------------------------------------------------- |
| gld_throughput     | l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second       |
| gst_throughput     | l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second       |
| gld_efficiency     | smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct |
| shared_efficiency  | smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct    |
| branch_efficiency  | smsp__sass_thread_inst_executed_op_control_pred_on.sum        |

多因一效
