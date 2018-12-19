//
//  active_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef active_tnna_h
#define active_tnna_h
#include <memory>
#include <vector>
#include "kernel_tnna.h"
namespace TNNA{
    template<typename Scale,typename Flow>
    class active{
    protected:
        Scale                       _rate;
		tensor<Flow>				_in;
		tensor<Flow>				_out;
        active(const Scale&rate=0.1):_rate(rate){}
    public:
		tensor<Flow>         in()const{ return _in; }
		tensor<Flow>&        in(){ return _in; }
		tensor<Flow>         out()const{ return _out; }
		tensor<Flow>&        out(){ return _out; }
        void                 clear(){_in=tensor<Flow>();_out=tensor<Flow>();}
		virtual void		 act(const size_t &itype, const size_t &otype, const size_t &nbat) = 0;
		virtual tensor<Flow> feedback(const size_t &itype, const size_t &otype, const size_t &nbat) = 0;
		virtual void print(std::ostream&ios) = 0;
    };
    template<typename Scale,typename Flow>
	class FunctionalActive :public active<Scale, Flow>
	{
	public:
		typedef std::shared_ptr<kernel<Flow>> Kernel;
	protected:
		std::valarray<Flow>						_args;
		Kernel                                  _kers;
		FunctionalActive(const Scale&rate = 0.1, const Kernel&kers = linearKernel<Flow>::New(), const std::valarray<Flow>& args = std::valarray<Flow>(2)) :active<Scale, Flow>(rate), _kers(kers), _args(args){}
    public:
		static std::shared_ptr<active<Scale, Flow>> New(const Scale&rate = 0.1, const Kernel&kers = linearKernel<Flow>::New(), const std::valarray<Flow>&  args = std::valarray<Flow>(2)){
			return std::shared_ptr<active<Scale, Flow>>(new FunctionalActive<Scale, Flow>(rate,kers, args));
		}
		virtual void	act(const size_t &itype, const size_t &otype, const size_t &nbat){
			if(!(this->_in.shape().size() <= 2 && this->_in.size() == itype*nbat))return;
			this->_in = tensor<Flow>({ itype, nbat }, this->_in.data());
			std::valarray<autodiff<Flow>> argsin(_args.size() + itype), argsout;
			for (size_t i = 0; i < _args.size() + itype; i++){
				if (i<_args.size()&&!_args[i].same(this->_in[{ 0, 0 }]))
					_args[i] = Flow(this->_in[{ 0, 0 }].shape(), 1.0, 1);
				std::valarray<Flow> Cells(_args.size() + itype);
				for (size_t j = 0; j < _args.size() + itype; j++)
					Cells[j] = Flow(this->_in[{ 0, 0 }].shape(), i == j ? 1.0 : 0.0, 0);
				argsin[i] = autodiff<Flow>(i<_args.size()?_args[i]:Flow(), Cells);
			}
			this->_out.resize({ otype, nbat });
			for (size_t n = 0; n<nbat; n++){
				for (size_t i = 0; i < itype;i++)
					argsin[_args.size() + i]._val = this->_in[{i, n}];
				argsout = _kers->apply(itype,otype,argsin);
				for (size_t o = 0; o < otype; o++)
					this->_out[{  o, n  }] = argsout[o]._val;
			}
        }
		virtual tensor<Flow> feedback(const size_t &itype, const size_t &otype, const size_t &nbat){
			assert(this->_in.shape().size() <= 2 && this->_in.size() == itype*nbat);
			assert(this->_out.shape().size() <= 2 && this->_out.size() == otype*nbat);
			this->_in = tensor<Flow>({ itype, nbat }, this->_in.data());
			this->_out = tensor<Flow>({ otype, nbat }, this->_out.data());
			std::valarray<autodiff<Flow>> argsin(_args.size() + itype), argsout;
			for (size_t i = 0; i < _args.size() + itype; i++){
				if (i<_args.size() && !_args[i].same(this->_in[{0, 0}]))
					_args[i] = Flow(this->_in[{0, 0}].shape(), 1.0, 1);
				std::valarray<Flow> Cells(_args.size() + itype);
				for (size_t j = 0; j < _args.size() + itype; j++)
					Cells[j] = Flow(this->_in[{ 0, 0 }].shape(), i == j ? 1.0 : 0.0, 0);
				argsin[i] = autodiff<Flow>(i<_args.size() ? _args[i] : Flow(), Cells);
			}
			tensor<Flow> outerr({ otype, nbat }, {});
			tensor<Flow> dydx({ otype, _args.size() + itype, nbat }, {});
			for (size_t n = 0; n<nbat; n++){
				for (size_t i = 0; i < itype;i++)
					argsin[_args.size() + i]._val = this->_in[{i, n}];
				argsout = _kers->apply(itype, otype, argsin);
				for (size_t o = 0; o < otype;o++)
				{
					outerr[{  o, n  }] = this->_out[{o, n}] - argsout[o]._val;
					dydx(std::map<size_t,size_t>({ { 0, o }, { 2, n } }), argsout[o]._dval);
				}
			}
			tensor<Flow> inerr({ itype, nbat }, std::valarray<Flow>(Flow(this->_in[{0, 0}].shape(), 0.0, 1), itype*nbat));
			// darg= rate*(dout/darg)'*outerr;
			// inerr=(dout/din)'*outerr;
			Flow rate(this->_in[{0, 0}].shape(), this->_rate, 0);
			for (size_t n = 0; n < nbat; n++)
			for (size_t o = 0; o < otype; o++)
			{
				for (size_t a = 0; a < _args.size(); a++)
					_args[a] = _args[a] + rate/nbat*dydx[{o, a, n}] * outerr[{o, n}];
				for (size_t i = 0; i < itype; i++)
					inerr[{i, n}] = inerr[{i, n}] + dydx[{o, i + _args.size(), n}] * outerr[{o, n}];
			}
			this->_in = this->_in + inerr;
            /*
			for (size_t a = 0; a < _args.size(); a++){
				_args[a](_args[a]>1e3, otype*tanh(_args[a]));
				_args[a](_args[a] < -1e3, otype*tanh(_args[a]));
				_args[a](isnan(_args[a]), 0.0);
				_args[a](isunnormal(_args[a]), 0.0);
			}
			for (size_t n = 0; n < nbat; n++)
			{
				for (size_t i = 0; i < itype; i++)
				{
					this->_in[{i, n}](this->_in[{i, n}]>1e3, otype*tanh(this->_in[{i, n}]));
					this->_in[{i, n}](this->_in[{i, n}] < -1e3, otype*tanh(this->_in[{i, n}]));
					this->_in[{i, n}](isnan(this->_in[{i, n}]), 0.0);
					this->_in[{i, n}](isunnormal(this->_in[{i, n}]), 0.0);
				}
			}
            */
			return outerr;
        }
		virtual void print(std::ostream&ios){
			ios << "active:{FunctionalActive,";
			ios << "Rate:" << this->_rate << ",";
			ios << "Args:{";
			for (int i = 0; i < (int)_args.size() - 1; i++)
				ios << _args[i] << ",";
            if(_args.size()>0)ios << _args[_args.size() - 1];
            ios<< "};";
			_kers->print(ios);
			ios << "};";
		}
     };
}
#endif /* active_h */
