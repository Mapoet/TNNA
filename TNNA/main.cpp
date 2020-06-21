//
//  main.cpp
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/2.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//
#include <fstream>
#include "tnna.h"
int main(int argc, const char * argv[]) {
	// insert code here...
	typedef std::chrono::milliseconds msec;
	using namespace TNNA;

	typedef tensor<double> Tensor;
	//	typedef double Tensor;
#define initTensor(x,type)   Tensor({4,4}, x, type)
#define initRandom			 initTensor(1.0/std::sqrt(7.0),0)
	double rate = 0.005;
	typedef point<3, double> Geometry;
	typedef graph <double, Tensor, Geometry > Graph;
	

#define tlinear (FunctionalTransmit<double, Tensor>::New(rate, functionKernel<Tensor>::New("tlinear",\
[](const size_t&itype, const size_t&otype, const std::valarray<autodiff<Tensor>>& argsin){\
std::valarray<autodiff<Tensor>> argsout(otype);\
Tensor zeros(argsin[2 + 0]._val.shape(), 0.0, 0);\
autodiff<Tensor> sums(zeros, std::valarray<Tensor>(zeros, argsin[2 + 0]._dval.size()));\
for (size_t i = 0; i < itype; i++){\
sums = sums + argsin[2 + i];\
}\
for (size_t i = 0; i < otype; i++){\
argsout[i] = argsin[0] + argsin[1] * sums;\
argsout[i]=atan(argsout[i]);\
}\
return argsout;\
}), { initRandom, initRandom }))
#define alinear (FunctionalActive<double, Tensor>::New(rate, functionKernel<Tensor>::New("alinear",\
        [](const size_t&itype, const size_t&otype, const std::valarray<autodiff<Tensor>>& argsin){\
            std::valarray<autodiff<Tensor>> argsout(otype);\
            Tensor zeros(argsin[2 + 0]._val.shape(), 0.0, 0);\
            autodiff<Tensor> sums(zeros, std::valarray<Tensor>(zeros, argsin[2 + 0]._dval.size()));\
            for (size_t i = 0; i < itype; i++){\
                sums = sums + argsin[2 + i];\
            }\
            for (size_t i = 0; i < otype; i++){\
                argsout[i] = argsin[0] + argsin[1] * sums;\
                argsout[i]=atan(argsout[i]);\
            }\
           return argsout;\
        }), { initRandom, initRandom }))
#define linear6 (FunctionalActive<double, Tensor>::New(rate, functionKernel<Tensor>::New("linear6",\
[](const size_t&itype, const size_t&otype, const std::valarray<autodiff<Tensor>>& argsin){\
std::valarray<autodiff<Tensor>> argsout(otype);\
Tensor zeros(argsin[3 + 0]._val.shape(), 0.0, 0);\
autodiff<Tensor> sums(zeros, std::valarray<Tensor>(zeros, argsin[2 + 0]._dval.size()));\
for (size_t i = 0; i < itype; i++){\
sums = sums + argsin[2 + i];\
}\
for (size_t i = 0; i < otype; i++){\
argsout[i] = argsin[0] + argsin[1+i] * sums;\
argsout[i]=atan(argsout[i]);\
}\
return argsout;\
}), { initRandom,initRandom,initRandom, initRandom,initRandom,initRandom, initRandom }))
#define weaklinear FunctionalActive<double, Tensor>::New(rate, weaklinearKernel<Tensor>::New(), { initRandom, initRandom })
	Graph::GRAPH gs=Graph::New();
	{
		Graph::Nodes nodes;
		Graph::Links links;
		Graph::LabelIOStream ios;
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), linear6);
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), weaklinear);
        nodes.emplace_back(DataValue <Geometry>::New(Geometry()), alinear);
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), alinear);
        // {0}->{1,2,3,4,5,6}
        for(size_t i=0;i<6;i++)
            links.emplace_back(0, 1+i, tlinear);
        //{1,2,3,4,5,6}->{7,8,9,10}
        for(size_t i=0;i<6;i++)for(size_t j=0;j<4;j++)
            links.emplace_back(1+i, 7+j, tlinear);
        //{7,8,9,10}->{11,12,13,14,15}
        for(size_t i=0;i<4;i++)for(size_t j=0;j<5;j++)
		links.emplace_back(7+i, 11+j, tlinear);
        //{11,12,13,14,15}->{16,17}
        for(size_t i=0;i<5;i++)for(size_t j=0;j<2;j++)
            links.emplace_back(11+i, 16+j, tlinear);
		ios.emplace_back(cellStream_Input, 0, DataStream<Tensor>::New(initRandom));
		ios.emplace_back(cellStream_Output, 16, DataStream<Tensor>::New(initRandom));
        ios.emplace_back(cellStream_Output, 17, DataStream<Tensor>::New(initRandom));
		gs->BuildStruct(nodes, links, ios);
	}
	size_t nbat = 500;
	gs->StartCell(nbat, msec(10));

	tensor<Tensor> xdata({ 1, nbat }, {});
	tensor<Tensor> ydata({ 2, nbat }, {});
	Tensor a = initTensor(0.8, 0);
	Tensor b = initTensor(0.5, 0);
    Tensor c = initTensor(0.6, 0);
    Tensor d = initTensor(0.4, 0);
	std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
	for (int i = 0; i < 200; i++)
	{
		for (size_t j = 0; j < nbat; j++){ 
			xdata[{0, j}] = initTensor(1.0*j, 1);
			ydata[{0, j}] = a + b*xdata[{0, j}];
            ydata[{1, j}] = c + d*xdata[{0, j}];
		}
		gs->Learning(xdata, ydata,false, msec(2000));
	}
	std::chrono::steady_clock::time_point et = std::chrono::steady_clock::now();
	gs->print(std::cout);
	nbat=5;
	gs->ReSetBat(nbat);
	xdata.resize({1, nbat});
	ydata.resize({2, nbat});
	for (size_t j = 0; j < nbat; j++){
		xdata[{0, j}] = initTensor(1.0*j, 1);
		ydata[{0, j}] = a + b*xdata[{0, j}];
        ydata[{1, j}] = c + d*xdata[{0, j}];
	}
	tensor<Tensor> rdata;
	std::chrono::steady_clock::time_point sl = std::chrono::steady_clock::now();
	gs->Thinking(xdata, rdata,false, msec(2000));
	auto node = Graph::Generate(gs.get(), Geometry(), linearKernel<Tensor>::New(), {initRandom, initRandom});
	node->data()[{0}]=0.2;
	node->data()[{1}]=0.3;
	gs->Get(0)->insertO(node, tlinear);
	std::ofstream out("test.dat");
    gs->print(out);
	out.close();
	gs->Remove(node);
	
	
	std::chrono::steady_clock::time_point el = std::chrono::steady_clock::now();
	std::cout << "final:\n" << ydata - rdata<<std::endl;
	std::cout << "Learning:" << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1e3 << std::endl;
	std::cout << "Thinking:" << std::chrono::duration_cast<std::chrono::milliseconds>(el - sl).count() / 1e3 << std::endl;
	return 0;
}

