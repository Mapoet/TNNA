//
//  autodiff_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/2.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef autodiff_tnna_h
#define autodiff_tnna_h
#include <valarray>
#include "tensor_tnna.h"
namespace TNNA{
    template<class Cell>
    struct autodiff{
        Cell   _val;
        std::valarray<Cell>   _dval;
        autodiff(const Cell& val = Cell(), std::valarray<Cell> dval = std::valarray<Cell>()) :_val(val), _dval(dval){}
        autodiff<Cell> operator=(const Cell& v){
            return autodiff<Cell>(v);
        }
        operator Cell()const{
            return _val;
        }
        autodiff<Cell> operator +=(const autodiff<Cell>& a){
            _val =_val+ a._val;
            _dval =_dval+ a._dval;
            return *this;
        }
        autodiff<Cell> operator -=(const autodiff<Cell>& a){
			_val = _val- a._val;
			_dval = _dval- a._dval;
            return *this;
        }
        autodiff<Cell> operator *=(const autodiff<Cell>& a){
            auto  val = _val;
            auto dval = _dval;
            _val = val*a._val;
            _dval = dval*a._val + val*a._dval;
            return *this;
        }
        autodiff<Cell> operator /=(const autodiff<Cell>& a){
            auto  val = _val;
            auto dval = _dval;
            _val = val / a._val;
            _dval = (dval*a._val - val*a._dval) / (a._val*a._val);
            return *this;
        }
    };
    template<class Cell>
    autodiff<Cell> operator+(const autodiff<Cell>& a, const autodiff<Cell>& b){
        autodiff<Cell> v = a;
        v += b;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator-(const autodiff<Cell>& a, const autodiff<Cell>& b){
        autodiff<Cell> v = a;
        v -= b;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator*(const autodiff<Cell>& a, const autodiff<Cell>& b){
        autodiff<Cell> v = a;
        v *= b;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator/(const autodiff<Cell>& a, const autodiff<Cell>& b){
        autodiff<Cell> v = a;
        v /= b;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator+(Cell a, const autodiff<Cell>& b){
        autodiff<Cell> v =b;
        v += a;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator-(Cell a, const autodiff<Cell>& b){
        autodiff<Cell> v;
        v._val = a - b._val;
        v._dval = -b._dval;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator*(const Cell& a, const autodiff<Cell>& b){
        autodiff<Cell> v;
        v._val = a * b._val;
        v._dval = a*b._dval;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator/(const Cell& a, const autodiff<Cell>& b){
        autodiff<Cell> v;
        v._val = a / b._val;
        v._dval = -a*b._dval / (b._val*b._val);
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator+(const autodiff<Cell>& a, const Cell& b){
        autodiff<Cell> v = a;
		v._val = v._val+ b;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator-(const autodiff<Cell>& a, const Cell& b){
        autodiff<Cell> v = a;
		v._val = v._val- b;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator*(const autodiff<Cell>& a, const Cell& b){
        autodiff<Cell> v = a;
		v._val = v._val* b;
		v._dval = v._dval* b;
        return v;
    }
    template<class Cell>
    autodiff<Cell> operator/(const autodiff<Cell>& a, const Cell& b){
        autodiff<Cell> v = a;
		v._val = v._val/ b;
		v._dval = v._dval/ b;
        return v;
	}
	template<class Cell>
	autodiff<Cell> pow(const autodiff<Cell>& a, const autodiff<Cell>& b){
		autodiff<Cell> v;
		v._val = std::pow(a._val, b._val);
		v._dval = b._val*std::pow(a._val, b._val - 1)*a._dval + v._val*std::log(a._val)*b._dval;
		return v;
	}
	template<class Cell>
	autodiff<Cell> pow(const Cell& a, const autodiff<Cell>& b){
		autodiff<Cell> v;
		v._val = std::pow(a, b._val);
		v._dval = v._val*std::log(a)*b._dval;
		return v;
	}
	template<class Cell>
	autodiff<Cell> pow(const autodiff<Cell>& a, const Cell& b){
		autodiff<Cell> v;
		v._val = std::pow(a._val, b);
		v._dval = b*std::pow(a._val, b - 1)*a._dval;
		return v;
	}
	template<class Cell>
	autodiff<Cell>& abs(const autodiff<Cell>& a){
		autodiff<Cell> v;
		v._val = std::abs(a._val);
		v._dval = (a._val > 0 ? 1.0 : -1.0)*a._dval;
		return v;
	}
    template<class Cell>
    autodiff<Cell> sin(const autodiff<Cell>& a){
        autodiff<Cell> v;
        v._val = sin(a._val);
        v._dval = cos(a._val)*a._dval;
        return v;
    }
    template<class Cell>
    autodiff<Cell> cos(const autodiff<Cell>& a){
        autodiff<Cell> v;
        v._val = cos(a._val);
        v._dval = -sin(a._val)*a._dval;
        return v;
    }
    template<class Cell>
    autodiff<Cell> tan(const autodiff<Cell>& a){
        autodiff<Cell> v = sin(a) / cos(a);
        return v;
	}
	template<class Cell>
	autodiff<Cell> exp(const autodiff<Cell>& a){
		autodiff<Cell> v;
		v._val = std::exp(a._val);
		v._dval = std::exp(a._val)*a._dval;
		return v;
	}
	template<class Cell>
	autodiff<Cell> asin(const autodiff<Cell>& a){
		autodiff<Cell> v;
		v._val = asin(a._val);
		v._dval = 1.0/sqrt((1.0-a._val*a._val))*a._dval;
		return v;
	}
	template<class Cell>
	autodiff<Cell> acos(const autodiff<Cell>& a){
		autodiff<Cell> v;
		v._val = acos(a._val);
		v._dval = -1.0/sqrt((1.0-a._val*a._val))*a._dval;
		return v;
	}
	template<class Cell>
	autodiff<Cell> atan(const autodiff<Cell>& a){
        autodiff<Cell> v;
        v._val = atan(a._val);
        v._dval = -1.0/(a._val*a._val+1.0)*a._dval;
		return v;
	}
	template<class Cell>
	autodiff<Cell> log(const autodiff<Cell>& a){
		autodiff<Cell> v;
		v._val = std::log(a._val);
		v._dval = a._dval / (a._val);
		return v;
	}
}
#endif /* autodiff_h */
