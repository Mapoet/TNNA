//
//  main.cpp
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/2.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#include "tnna.h"
int main(int argc, const char * argv[]) {
	// insert code here...
	typedef std::chrono::milliseconds msec;
	using namespace TNNA;

	typedef tensor<double> Tensor;
	//	typedef double Tensor;
#define initTensor(x,type)   Tensor({5,5}, x, type)
#define initRandom			 initTensor(1.0/std::sqrt(7.0),0)
	double rate = 0.05;
	typedef point<3, double> Geometry;
	typedef graph <double, Tensor, Geometry > Graph;
	Graph gs;
	{
		Graph::Nodes nodes;
		Graph::Links links;
		Graph::LabelIOStream ios;
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), FunctionalActive<double, Tensor>::New(rate, linearKernel<Tensor>::New(), { initRandom, initRandom }));
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), FunctionalActive<double, Tensor>::New(rate, weaklinearKernel<Tensor>::New(), { initRandom, initRandom }));
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), FunctionalActive<double, Tensor>::New(rate, weaklinearKernel<Tensor>::New(), { initRandom, initRandom }));
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), FunctionalActive<double, Tensor>::New(rate, weaklinearKernel<Tensor>::New(), { initRandom, initRandom }));
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), FunctionalActive<double, Tensor>::New(rate, linearKernel<Tensor>::New(), { initRandom, initRandom }));
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), FunctionalActive<double, Tensor>::New(rate, linearKernel<Tensor>::New(), { initRandom, initRandom }));
		nodes.emplace_back(DataValue <Geometry>::New(Geometry()), FunctionalActive<double, Tensor>::New(rate, linearKernel<Tensor>::New(), { initRandom, initRandom }));
		links.emplace_back(0, 1, FunctionalTransmit<double, Tensor>::New(rate, weakluKernel<Tensor>::New(), { initRandom, initRandom }));
		links.emplace_back(0, 2, FunctionalTransmit<double, Tensor>::New(rate, weakluKernel<Tensor>::New(), { initRandom, initRandom }));
		links.emplace_back(0, 3, FunctionalTransmit<double, Tensor>::New(rate, weakluKernel<Tensor>::New(), { initRandom, initRandom }));
		links.emplace_back(1, 4, FunctionalTransmit<double, Tensor>::New(rate, reluKernel<Tensor>::New(), { initRandom }));
		links.emplace_back(1, 5, FunctionalTransmit<double, Tensor>::New(rate, reluKernel<Tensor>::New(), { initRandom }));
		links.emplace_back(2, 4, FunctionalTransmit<double, Tensor>::New(rate, reluKernel<Tensor>::New(), { initRandom }));
		links.emplace_back(2, 5, FunctionalTransmit<double, Tensor>::New(rate, reluKernel<Tensor>::New(), { initRandom }));
		links.emplace_back(3, 4, FunctionalTransmit<double, Tensor>::New(rate, reluKernel<Tensor>::New(), { initRandom }));
		links.emplace_back(3, 5, FunctionalTransmit<double, Tensor>::New(rate, reluKernel<Tensor>::New(), { initRandom }));
		links.emplace_back(4, 6, FunctionalTransmit<double, Tensor>::New(rate, weakluKernel<Tensor>::New(), { initRandom, initRandom }));
		links.emplace_back(5, 6, FunctionalTransmit<double, Tensor>::New(rate, weakluKernel<Tensor>::New(), { initRandom, initRandom }));
		ios.emplace_back(cellStream_Input, 0, DataStream<Tensor>::New(initRandom));
		ios.emplace_back(cellStream_Output, 6, DataStream<Tensor>::New(initRandom));
		gs.BuildStruct(nodes, links, ios);
	}
	size_t nbat = 30;
	gs.StartCell(nbat, msec(15));

	tensor<Tensor> xdata({ 1, nbat }, {});
	tensor<Tensor> ydata({ 1, nbat }, {});
	Tensor a = initTensor(0.8, 0);
	Tensor b = initTensor(0.5, 0);
	std::chrono::steady_clock::time_point st = std::chrono::steady_clock::now();
	for (int i = 0; i < 50; i++)
	{
		for (size_t j = 0; j < nbat; j++){ 
			xdata[{0, j}] = initTensor(1.0, 1);
			ydata[{0, j}] = a + b*xdata[{0, j}];
		}
		gs.Learning(xdata, ydata,false, msec(1000));
	}
	std::chrono::steady_clock::time_point et = std::chrono::steady_clock::now();
	gs.print(std::cout);
	nbat=50;
	gs.ReSetBat(nbat);
	xdata.resize({1, nbat});
	ydata.resize({1, nbat});
	for (size_t j = 0; j < nbat; j++){
		xdata[{0, j}] = initTensor(1.0, 1);
		ydata[{0, j}] = a + b*xdata[{0, j}];
	}
	tensor<Tensor> rdata;
	std::chrono::steady_clock::time_point sl = std::chrono::steady_clock::now();
	gs.Thinking(xdata, rdata,false, msec(1000));
	std::chrono::steady_clock::time_point el = std::chrono::steady_clock::now();
	std::cout << "final:\n" << ydata - rdata<<std::endl;
	std::cout << "Learning:" << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1e3 << std::endl;
	std::cout << "Thinking:" << std::chrono::duration_cast<std::chrono::milliseconds>(el - sl).count() / 1e3 << std::endl;
	return 0;
}

