//
//  kernel_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/2.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef kernel_tnna_h
#define kernel_tnna_h
#include <valarray>
#include "autodiff_tnna.h"
namespace TNNA{
	template<typename Cell>
	class kernel{
		template<typename Scale, typename Flow>
		friend class FunctionalActive;
		template<typename Scale, typename Flow>
		friend class FunctionalTransmit;
	protected:
		virtual std::valarray<autodiff<Cell>> argsinout(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin) = 0;
		virtual void print(std::ostream&ios) = 0;
		kernel(){}
	public:
		std::valarray<autodiff<Cell>>
			apply(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin){
				assert(itype <= argsin.size());
				return argsinout(itype,otype,argsin);
			}
	};
    template<typename Cell>
    class functionKernel :public kernel<Cell>{
        typedef std::valarray<autodiff<Cell>> (*Apply)(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin);
        Apply _apply;
        std::string _name;
    protected:
        virtual std::valarray<autodiff<Cell>> argsinout(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin){
            return _apply==nullptr?std::valarray<autodiff<Cell>>(otype):_apply(itype,otype,argsin);
        }
        virtual void print(std::ostream&ios){
            ios << "kernel:"<<_name;
        }
        functionKernel(const std::string&name="",const Apply& apply=nullptr):_name(name),_apply(apply){}
    public:
        static std::shared_ptr<kernel<Cell>> New(const std::string&name="",const Apply& apply=nullptr){
            return std::shared_ptr<kernel<Cell>>(new functionKernel(name,apply));
        }
    };
	template<typename Cell>
	class linearKernel :public kernel<Cell>{
	protected:
		virtual std::valarray<autodiff<Cell>> argsinout(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin){
			std::valarray<autodiff<Cell>> argsout(otype);
			Cell zeros(argsin[2 + 0]._val.shape(), 0.0, 0);
			autodiff<Cell> sums(zeros, std::valarray<Cell>(zeros, argsin[2 + 0]._dval.size()));
			for (size_t i = 0; i < itype; i++){
				sums = sums + argsin[2 + i];
			}
			for (size_t i = 0; i < otype; i++){
				argsout[i] = argsin[0] + argsin[1] * sums;
                //argsout[i]=atan(argsout[i]);
			}
			return argsout;
		}
		virtual void print(std::ostream&ios){
			ios << "kernel:linearKernel";
		}
		linearKernel(){}
	public:
		static std::shared_ptr<kernel<Cell>> New(){
			return std::shared_ptr<kernel<Cell>>(new linearKernel());
		}
	};
	template<typename Cell>
	class reluKernel :public kernel<Cell>{
	protected:
		virtual std::valarray<autodiff<Cell>> argsinout(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin){
			std::valarray<autodiff<Cell>> argsout(otype);
			Cell zeros(argsin[1 + 0]._val.shape(), 0.0, 0);
			autodiff<Cell> sums(zeros, std::valarray<Cell>(zeros, argsin[1 + 0]._dval.size()));
			for (size_t i = 0; i < itype; i++){
				sums = sums + argsin[1 + i];
			}
			for (size_t i = 0; i < otype; i++){
				argsout[i] = argsin[0] + sums;
				argsout[i]._val(argsout[i]._val < 0.0, 0.0);
				for (size_t j = 0; j < argsout[i]._dval.size(); j++)
					argsout[i]._dval[j](argsout[i]._val < 0.0, 0.0);
               // argsout[i]=atan(argsout[i]);
			}
			return argsout;
		}
		virtual void print(std::ostream&ios){
			ios << "kernel:reluKernel";
		}
		reluKernel(){}
	public:
		static std::shared_ptr<kernel<Cell>> New(){
			return std::shared_ptr<kernel<Cell>>(new reluKernel());
		}
	};
	template<typename Cell>
	class weakluKernel :public kernel<Cell>{
	protected:

		virtual std::valarray<autodiff<Cell>> argsinout(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin){
			std::valarray<autodiff<Cell>> argsout(otype);
			Cell zeros(argsin[2 + 0]._val.shape(), 0.0, 0);
			autodiff<Cell> sums(zeros, std::valarray<Cell>(zeros, argsin[2 + 0]._dval.size()));
			for (size_t i = 0; i < itype; i++){
				sums = sums + argsin[2 + i];
			}
			for (size_t i = 0; i < otype; i++){
				argsout[i] = argsin[0] + argsin[1]*sums;
				auto is = (argsout[i]._val < 0.0);
				argsout[i]._val(is, 0.0);
				for (size_t j = 0; j < argsout[i]._dval.size(); j++)
					argsout[i]._dval[j](is, 0.0);
                //argsout[i]=atan(argsout[i]);
			}
			return argsout;
		}
		virtual void print(std::ostream&ios){
			ios << "kernel:weakluKernel";
		}
		weakluKernel(){}
	public:
		static std::shared_ptr<kernel<Cell>> New(){
			return std::shared_ptr<kernel<Cell>>(new weakluKernel());
		}
	};

	template<typename Cell>
	class weaklinearKernel :public kernel<Cell>{
	protected:
		virtual std::valarray<autodiff<Cell>> argsinout(const size_t&itype, const size_t&otype, const std::valarray<autodiff<Cell>>& argsin){
			std::valarray<autodiff<Cell>> argsout(otype);
			Cell zeros(argsin[2 + 0]._val.shape(), 0.0, 0);
			autodiff<Cell> sums(zeros, std::valarray<Cell>(zeros, argsin[2 + 0]._dval.size()));
			for (size_t i = 0; i < itype; i++){
				sums = sums + argsin[2 + i];
			}
			for (size_t i = 0; i < otype; i++){
				argsout[i] = argsin[0] + argsin[1] * sums;
				auto is = (argsout[i]._val > 0.0);
				argsout[i]._val(is, argsin[0]._val + sums._val);
				for (size_t j = 0; j < argsout[i]._dval.size(); j++)
					argsout[i]._dval[j](is, 1.0);
               // argsout[i]=atan(argsout[i]);
			}
			return argsout;
		}
		virtual void print(std::ostream&ios){
			ios << "kernel:weaklinearKernel";
		}
		weaklinearKernel(){}
	public:
		static std::shared_ptr<kernel<Cell>> New(){
			return std::shared_ptr<kernel<Cell>>(new weaklinearKernel());
		}
	};
}
#endif
