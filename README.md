# TNNA（基于张量图结构的元胞自动机）
## 简介
    通过图结构与元胞自动机机制，构建具有自适应及自调整型的神经网络结构。
    以元胞作为基本框架（通常以用户指定型函数作为运算方式），通过拓扑链接（通常以加权函数作为形式）形成元胞邻域，进而构建出混合型张量流传递图结构（包括树状结构与环状结构），最终可以实现功能型网络结构及状态机型网络结构，而具有：
        数据->信息->识别->判断->信息->控制
    的流程。