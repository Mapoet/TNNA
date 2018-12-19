//
//  transmit_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef transmit_tnna_h
#define transmit_tnna_h
#include <memory>
#include <vector>
#include "kernel_tnna.h"
namespace TNNA{
    template<typename Scale,typename Flow>
    class transmit{
    protected:
        Scale                       _rate;
        transmit(const Scale&rate=0.1):_rate(rate){}
    public:
        virtual tensor<Flow> predict(const tensor<Flow>& inputs)=0;
		virtual tensor<Flow>  update(const tensor<Flow>& inputs, const tensor<Flow>& outputs) = 0;
		virtual void print(std::ostream&ios) = 0;
    };
	template<typename Scale, typename Flow>
	class FunctionalTransmit :public transmit<Scale, Flow>
	{
	  public:
		typedef std::shared_ptr<kernel<Flow>> Kernel;
	  protected:
		std::valarray<Flow> _args;
		Kernel _kers;
		FunctionalTransmit(const Scale&rate = 0.1, const Kernel&kers = linearKernel<Flow>::New(), const std::valarray<Flow>&args = std::valarray<Flow>(2)) :transmit<Scale,Flow>(rate),_kers(kers), _args(args){}
	public:
		static std::shared_ptr<transmit<Scale, Flow>> New(const Scale&rate = 0.1, const Kernel&kers = linearKernel<Flow>::New(), const std::valarray<Flow>& args = std::valarray<Flow>(2)){
			return std::shared_ptr<transmit<Scale, Flow>>(new FunctionalTransmit<Scale, Flow>(rate,kers, args));
		}
		virtual tensor<Flow> predict(const tensor<Flow>& inputs){
			assert(inputs.size()>0&&inputs.shape().size() <= 1);
			size_t nbat = inputs.shape().size() == 0 ? 1 : inputs.shape()[0];
			std::valarray<autodiff<Flow>> argsin(_args.size() + 1), argsout;
			for (size_t i = 0; i < _args.size() + 1; i++){
				if (i<_args.size() && !_args[i].same(inputs[{0}]))
					_args[i] = Flow(inputs[{0}].shape(), 1.0, 1);
				std::valarray<Flow> Cells(_args.size() + 1);
				for (size_t j = 0; j < _args.size() + 1; j++)
					Cells[j] = Flow(inputs[{ 0 }].shape(), i == j ? 1.0 : 0.0, 0);
				argsin[i] = autodiff<Flow>(i<_args.size()?_args[i]:Flow(), Cells);
			}
			tensor<Flow> out({ nbat }, std::valarray<Flow>());
			for (size_t n = 0; n<nbat; n++){
				argsin[_args.size()]._val = inputs[{n}];
				out[{n}] = _kers->apply(1, 1, argsin)[0]._val;
			}
			return out;
		}
		virtual tensor<Flow> update(const tensor<Flow>& inputs, const tensor<Flow>& outputs){
			assert(inputs.size() > 0 && inputs.size() == outputs.size());
			assert(inputs.shape().size() <= 1 && outputs.shape().size() <= 1);
			size_t nbat = inputs.shape().size() == 0 ? 1 : inputs.shape()[0];
			std::valarray<autodiff<Flow>> argsin(_args.size() + 1), argsout;
			for (size_t i = 0; i < _args.size() + 1; i++){
				if (i<_args.size() && !_args[i].same(inputs[{0}]))
					_args[i] = Flow(inputs[{0}].shape(), 1.0, 1);
				std::valarray<Flow> Cells(_args.size() + 1);
				for (size_t j = 0; j < _args.size() + 1; j++)
					Cells[j] = Flow(inputs[{ 0 }].shape(), i == j ? 1.0 : 0.0, 0);
				argsin[i] = autodiff<Flow>(i<_args.size() ? _args[i] : Flow(), Cells);
			}
			tensor<Flow> outerr({ nbat }, {});
			tensor<Flow> dydx({ _args.size() + 1, nbat }, {});
			for (size_t n = 0; n<nbat; n++){
				argsin[_args.size()]._val = inputs[{n}];
				argsout = _kers->apply(1, 1, argsin);
				outerr[{n}] = outputs[{n}] -argsout[0]._val;
				dydx(std::map<size_t,size_t>({ { 1, n } }), argsout[0]._dval);
			}
			tensor<Flow> inerr({ nbat }, {});
			// darg= rate*(dout/darg)'*outerr;
			// inerr=(dout/din)'*outerr;
			Flow rate(inputs[{ 0 }].shape(), this->_rate, 0);
			for (size_t n = 0; n < nbat; n++){
				for (size_t a = 0; a < _args.size(); a++)
					_args[a] = _args[a] + rate/nbat*dydx[{a, n}] * outerr[{n}] ;
				inerr[{n}] = dydx[{_args.size(), n}] * outerr[{n}];
			}
            /*
			for (size_t a = 0; a < _args.size(); a++){
				_args[a](_args[a]>1e3,tanh(_args[a]));
				_args[a](_args[a] < -1e3,tanh(_args[a]));
				_args[a](isnan(_args[a]), 0.0);
				_args[a](isunnormal(_args[a]), 0.0);
			}
			for (size_t n = 0; n < nbat; n++)
			{
				inerr[{n}](inerr[{n}]>1e5, tanh(inerr[{n}]));
				inerr[{n}](inerr[{n}] < -1e5, tanh(inerr[{n}]));
				inerr[{n}](isnan(inerr[{n}]), 0.0);
				inerr[{n}](isunnormal(inerr[{n}]), 0.0);
			}
             */
			return inerr;
		}
		virtual void print(std::ostream&ios){
			ios << "transmit:{FunctionalTransmit,";
			ios << "Rate:" << this->_rate << ",";
			ios << "Args:{";
			for (size_t i = 0; i < _args.size() - 1; i++)
				ios << _args[i] << ",";
			ios << _args[_args.size() - 1] << "};";
			_kers->print(ios);
			ios << "};";
		}
	};
}

#endif /* transmit_h */
