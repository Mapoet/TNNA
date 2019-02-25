//
//  tensor_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/2.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef tensor_tnna_h
#define tensor_tnna_h
#include <valarray>
#include <vector>
#include <map>
#include <random>
#include <cassert>
#include <cmath>
namespace TNNA{
	template<typename Cell>
	class tensor;
    template<typename Cell>
    class tensor{
        size_t              _num;
        std::valarray<size_t> _shp;
        std::valarray<Cell> _val;
	public:
        static size_t idxs2loc(const std::valarray<size_t>& idxs,const std::valarray<size_t>& shp){
            size_t loc=0;
            if(idxs.size()==0)return loc;
            loc=idxs[0];
            for(size_t i=1;i<idxs.size();i++)
            if(idxs[i]>=shp[i])
                return -1;
            else
                loc=loc*shp[i]+idxs[i];
            return loc;
        }
        static std::valarray<size_t> loc2idxs(const size_t&loc,const std::valarray<size_t>& shp,const size_t& num){
            std::valarray<size_t> idxs;
            if(loc>num)return idxs;
            size_t n=loc;
            idxs.resize(shp.size());
            for(int i=(int)(shp.size())-1;i>=0;i--){
                idxs[i]=n%shp[i];
                n=n/shp[i];
            }
            return idxs;
        }
	public:
		tensor() :_shp(), _val(), _num(0){}
        tensor(const Cell& cell):_shp(),_val(cell,1),_num(1){}
		tensor(const std::valarray<size_t>& shp, const std::valarray<Cell>& val = std::valarray<Cell>()) :_shp(shp), _val(val){
            _num=1;
            for(auto i:_shp)_num*=i;
			if (_val.size() != _num)
				_val.resize(_num);
        }
		tensor(const std::valarray<size_t>& shp, const Cell& val, const size_t&type) :_shp(shp){
			_num = 1;
            for(auto i:_shp)_num*=i;
			std::random_device den;
			std::mt19937 gen(den());
            std::uniform_real_distribution<Cell> uniform(Cell(-1.0),Cell(1.0));
            std::normal_distribution<Cell>       normal(Cell(0.0),Cell(1.0));
            if(type/2==0)
            {
                _val=std::valarray<Cell>(val,_num);
				if (type % 2 == 1)for (size_t i = 0; i<_num; i++)_val[i] = _val[i]*normal(gen);
            }
            else{
                std::valarray<size_t> idxs(1,_shp.size());
                _val.resize(_num);
                for(size_t i=0;i<_shp.min();i++)
                    _val[idxs2loc(idxs*i, _shp)]=val*(type%2==1?normal(gen):Cell(1.0));
            }
            
        }
        void resize(const std::valarray<size_t> shp){
            _shp=shp;
            _num=1;
            for(auto i:_shp)_num*=i;
            _val.resize(_num);
        }
        std::valarray<size_t> shape()const{return _shp;}
		size_t              size()const{ return _num; }
        std::valarray<Cell> data()const{return _val;}
        std::valarray<Cell>& data(){return _val;}
        tensor<Cell>operator=(const tensor<Cell>& one){
            _shp=one._shp;
            _num=one._num;
            _val=one._val;
			return *this;
        }
        template<typename Type>
		tensor<Cell>& operator=(const tensor<Type>&& one){
			_shp = one._shp;
			_num = one.size();
            if(_num!=0)
                _val.resize(_num);
            else
                _val=std::valarray<Cell>();
            for(size_t i=0;i<_num;i++)
                _val[i]=one._val[i];
            return *this;
        }
		bool reshape(const std::valarray<size_t> shp){
            int num=1;
            for(auto i:shp)num*=i;
            if(num!=_num)return false;
            _shp=shp;
            return true;
        }
        template<typename Type>
        bool same(const tensor<Type>& one)const{
            if(_shp.size()!=one._shp.size()||_num!=one._num)return false;
            for(size_t i=0;i<_shp.size();i++)
                if(_shp[i]!=one._shp[i])return false;
            return true;
        }
        tensor<Cell> slice(const std::vector<std::vector<size_t> >&idxs)const{
            size_t n=1,loc=0,m=0;
            std::valarray<size_t> shp0,shp,idx;
            std::valarray<Cell>   val;
            shp0.resize(_shp.size());
            for(size_t i=0;i<idxs.size(); i++)
            {
                if(idxs[i].size()!=1){
                    m++;
                    n*=idxs[i].size();
                }
                shp0[i]=idxs[i].size();
            }
            val.resize(n);
            for(size_t i=0;i<n;i++)
            {
                idx=loc2idxs(i,shp0,n);
                for(size_t j=0;j<idxs.size();j++)
                    idx[j]=idxs[j][idx[j]];
				loc = idxs2loc(idx, _shp);
                val[i]=_val[loc];
            }
            shp.resize(m);
            m=0;
            for(size_t i=0;i<idxs.size(); i++)
            if(idxs[i].size()!=1){
				shp[m] = idxs[i].size();
				m++;
            }
            return tensor<Cell>(shp, val);
        }
        bool slice(const std::vector<std::vector<size_t> >&idxs,const std::valarray<Cell>&val){
            size_t n=1,loc=0;
            std::valarray<size_t> shp0,shp,idx;
            shp0.resize(_shp.size());
			for (size_t i = 0; i < idxs.size(); i++)
			{
				shp0[i] = idxs[i].size();
				n *= shp0[i];
			}
			assert(val.size() == n);
            for(size_t i=0;i<n;i++)
            {
                idx=loc2idxs(i,shp0,n);
                for(size_t j=0;j<idxs.size();j++)
                    idx[j]=idxs[j][idx[j]];
                loc=idxs2loc(idx, _shp);
                if(loc>_num)return false;
                _val[loc]=val[i];
            }
            return true;
        }
        bool insert(const std::pair<size_t,size_t>&loc,const tensor<Cell>&block){

            return true;
        }
        bool remove(const std::pair<size_t,size_t>&loc){
            
            return true;
        }
        Cell each(const size_t &loc)const{
            return (int)loc==-1||loc>=_num?Cell():_val[loc];
        }
        Cell& each(const size_t &loc){
            assert((int)loc!=-1);
            return _val[loc];
        }
        Cell operator[](const std::valarray<size_t> &idxs)const{
            size_t loc=idxs2loc(idxs,_shp);
            return (int)loc==-1||loc>=_num?Cell():_val[loc];
        }
        Cell& operator[](const std::valarray<size_t> &idxs){
            size_t loc=idxs2loc(idxs,_shp);
            assert((int)loc!=-1);
            return _val[loc];
        }
        tensor<Cell> operator()(const std::map<size_t,std::vector<size_t> >&rage)const{
            std::vector<std::vector<size_t> > idxs;
            size_t n=0;
            idxs.resize(_shp.size());
            for(auto it:rage){
                it.second[2]=it.second[2]==-1?_shp[it.first]:it.second[2];
                if(it.second[0]>_shp[it.first]||it.second[2]<it.second[0])
                    return tensor<Cell>();
                n=(it.second[2]-it.second[0])/it.second[1];
                idxs.at(it.first).resize(n);
                for(size_t i=0;i<n;i++)
                    idxs[it.first][i]=it.second[0]+i*it.second[1];
            }
            for(size_t i=0;i<_shp.size();i++)if(idxs[i].size()==0)
            {
                idxs[i].resize(_shp[i]);
                for(size_t j=0;j<_shp[i];j++)
                    idxs[i][j]=j;
            }
            return slice(idxs);
        }
        tensor<Cell> operator()(const std::map<size_t,size_t>&rage)const{
            std::vector<std::vector<size_t> > idxs;
            idxs.resize(_shp.size());
            for(auto it:rage){
                idxs.at(it.first).resize(1);
                idxs[it.first][0]=it.second;
            }
            for(size_t i=0;i<_shp.size();i++)if(idxs[i].size()==0)
            {
                idxs[i].resize(_shp[i]);
                for(size_t j=0;j<_shp[i];j++)
                    idxs[i][j]=j;
            }
            return slice(idxs);
		}
		tensor<Cell> operator()(const tensor<bool>&sels)const{
			assert(_num == sels.size());
			tensor<Cell> rt(_shp, std::valarray<Cell>());
			for (size_t i = 0; i < _num; i++)rt._val[i] = sels.data()[i] ? _val[i] : Cell();
			return rt;
		}
		bool operator()(const std::map<size_t, std::vector<size_t> >&rage, const std::valarray<Cell>&val){
			std::vector<std::vector<size_t> > idxs;
			size_t n = 0;
			idxs.resize(_shp.size());
			for (auto it : rage){
				it.second[2] = it.second[2] == -1 ? _shp[it.first] : it.second[2];
				if (it.second[0]>_shp[it.first] || it.second[2]<it.second[0])
					return tensor<Cell>();
				n = (it.second[2] - it.second[0]) / it.second[1];
				idxs.at(it.first).resize(n);
				for (size_t i = 0; i<n; i++)
					idxs[it.first][i] = it.second[0] + i*it.second[1];
			}
			for (size_t i = 0; i<_shp.size(); i++)if (idxs[i].size() == 0)
			{
				idxs[i].resize(_shp[i]);
				for (size_t j = 0; j<_shp[i]; j++)
					idxs[i][j] = j;
			}
			return slice(idxs,val);
		}
		bool operator()(const std::map<size_t, size_t>&rage, const std::valarray<Cell>&val){
			std::vector<std::vector<size_t> > idxs;
			idxs.resize(_shp.size());
			for (auto it : rage){
				idxs.at(it.first).resize(1);
				idxs[it.first][0] = it.second;
			}
			for (size_t i = 0; i<_shp.size(); i++)if (idxs[i].size() == 0)
			{
				idxs[i].resize(_shp[i]);
				for (size_t j = 0; j<_shp[i]; j++)
					idxs[i][j] = j;
			}
			return slice(idxs, val);
		}
		bool operator()(const tensor<bool>&sels,const Cell&val){
			if(_num != sels.size())return false;
			for (size_t i = 0; i < _num; i++)_val[i] = sels.data()[i] ? val : _val[i];
			return true;
		}
		bool operator()(const tensor<bool>&sels, Cell(*_func)(const Cell&)){
			if (_num != sels.size())return false;
			for (size_t i = 0; i < _num; i++)_val[i] = sels.data()[i] ? _func(_val[i]) : _val[i];
			return true;
		}
		bool operator()(const tensor<bool>&sels, const tensor<Cell>&block){
			if (_num != sels.size())return false;
			for (size_t i = 0; i < _num; i++)_val[i] = sels.data()[i] ? block._val[i] : _val[i];
			return true;
		}
        tensor<Cell> operator +()const{return *this;}
        tensor<Cell> operator -()const{
            tensor<Cell> one=*this;
            one._val=one._val*Cell(-1);
            return one;
        }
        tensor<Cell> apply(Cell(*_Apply)(const Cell&))const{
            return tensor<Cell>(_shp,_val.apply(_Apply));
        }
		friend std::ostream& operator<<(std::ostream&ios, const tensor<Cell>&ts){
			ios << "tensor:{num:" << ts._num << ",";
			if (ts._shp.size() > 0)
			{
				ios << "shape:{";
				for (size_t i = 0; i < ts._shp.size()-1; i++)
					ios << ts._shp[i] << ",";
				ios << ts._shp[ts._shp.size() - 1] << "},";
				ios << "val:{";
				for (size_t i = 0; i < ts._num-1; i++)
					ios << ts._val[i] << ",";
				ios << ts._val[ts._num - 1] << "}}";
			}
			else
			{
				if (ts._num == 1)
					ios << "val:" << ts._val[0] << "}";
                    else
                        ios << "val:{}}";
            }
			return ios;
		}
        friend tensor<Cell> sin(const tensor<Cell> &a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = sin(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell> cos(const tensor<Cell> &a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = cos(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell> tan(const tensor<Cell> &a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = tan(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell> exp(const tensor<Cell> &a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = exp(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell> asin(const tensor<Cell> &a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = asin(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell> acos(const tensor<Cell> &a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = atan(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell>atan(const tensor<Cell> & a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = atan(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
		friend tensor<Cell>tanh(const tensor<Cell> & a){
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = tanh(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell>sqrt(const tensor<Cell> & a){
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = sqrt(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<Cell> log(const tensor<Cell> &a)
        {
            using namespace std;
            std::valarray<Cell> val(a._num);
            for (size_t i = 0; i < a._num; i++)
                val[i] = log(a._val[i]);
            return tensor<Cell>(a._shp, val);
        }
        friend tensor<bool>isnan(const tensor<Cell> & a){
			using namespace std;
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)
				val[i] = std::isnan(a._val[i]);
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool>isfinite(const tensor<Cell> & a){
			using namespace std;
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = std::isfinite(a._val[i]);
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool>isnormal(const tensor<Cell> & a){
			using namespace std;
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = std::isnormal(a._val[i]);
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool>isunnormal(const tensor<Cell> & a){
			using namespace std;
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = !std::isnormal(a._val[i]);
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator > (const tensor<Cell> & a, const tensor<Cell> & b){
			if (a.size()==0||!a.same(b))return tensor<bool>();
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] > b._val[i];
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator >= (const tensor<Cell> & a, const tensor<Cell> & b){
			if (a.size()==0||!a.same(b))return tensor<bool>();
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] >= b._val[i];
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator < (const tensor<Cell> & a, const tensor<Cell> & b){
			if (a.size()==0||!a.same(b))return tensor<bool>();
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] < b._val[i];
			return tensor<bool>(a._shp,val);
		}
		friend tensor<bool> operator <= (const tensor<Cell> & a, const tensor<Cell> & b){
			if (a.size()==0||!a.same(b))return tensor<bool>();
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] <= b._val[i];
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator == (const tensor<Cell> & a, const tensor<Cell> & b){
			if (a.size()==0||!a.same(b))return tensor<bool>();
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] == b._val[i];
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator != (const tensor<Cell> & a, const tensor<Cell> & b){
			if (a.size()==0||!a.same(b))return tensor<bool>();
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] != b._val[i];
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator >(const Cell & a, const tensor<Cell> & b){
			std::valarray<bool >val(b._num);
			for (size_t i = 0; i<b._num; i++)val[i] = a > b._val[i];
			return tensor<bool>(b._shp, val);
		}
		friend tensor<bool> operator >= (const Cell & a, const tensor<Cell> & b){
			std::valarray<bool >val(b._num);
			for (size_t i = 0; i<b._num; i++)val[i] = a >= b._val[i];
			return tensor<bool>(b._shp, val);
		}
		friend tensor<bool> operator < (const Cell & a, const tensor<Cell> & b){
			std::valarray<bool >val(b._num);
			for (size_t i = 0; i<b._num; i++)val[i] = a < b._val[i];
			return tensor<bool>(b._shp, val);
		}
		friend tensor<bool> operator <= (const Cell & a, const tensor<Cell> & b){
			std::valarray<bool >val(b._num);
			for (size_t i = 0; i<b._num; i++)val[i] = a <= b._val[i];
			return tensor<bool>(b._shp, val);
		}
		friend tensor<bool> operator == (const Cell & a, const tensor<Cell> & b){
			std::valarray<bool >val(b._num);
			for (size_t i = 0; i<b._num; i++)val[i] = a == b._val[i];
			return tensor<bool>(b._shp, val);
		}
		friend tensor<bool> operator != (const Cell & a, const tensor<Cell> & b){
			std::valarray<bool >val(b._num);
			for (size_t i = 0; i<b._num; i++)val[i] = a != b._val[i];
			return tensor<bool>(b._shp, val);
		}
		friend tensor<bool> operator >(const tensor<Cell> & a, const Cell & b){
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] > b;
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator >= (const tensor<Cell> & a, const Cell & b){
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] >= b;
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator < (const tensor<Cell> & a, const Cell & b){
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] < b;
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator <= (const tensor<Cell> & a, const Cell & b){
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] <= b;
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator == (const tensor<Cell> & a, const Cell & b){
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] == b;
			return tensor<bool>(a._shp, val);
		}
		friend tensor<bool> operator != (const tensor<Cell> & a, const Cell & b){
			std::valarray<bool >val(a._num);
			for (size_t i = 0; i<a._num; i++)val[i] = a._val[i] != b;
			return tensor<bool>(a._shp, val);
		}
        friend tensor<Cell> operator +(const tensor<Cell> & a,const tensor<Cell> & b){
            tensor<Cell> rt;
            if(a.size()==0||!a.same(b))return rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val+b._val;
            return rt;
        }
        friend tensor<Cell> operator -(const tensor<Cell> & a,const tensor<Cell> & b){
            tensor<Cell> rt;
            if(a.size()==0||!a.same(b))return rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val-b._val;
            return rt;
        }
        friend tensor<Cell> operator *(const tensor<Cell> & a,const tensor<Cell> & b){
            tensor<Cell> rt;
            if(a.size()==0||!a.same(b))return rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val*b._val;
            return rt;
        }
		friend tensor<Cell> operator /(const tensor<Cell> & a, const tensor<Cell> & b){
            tensor<Cell> rt;
            if(a.size()==0||!a.same(b))return rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val/b._val;
            return rt;
        }
		friend tensor<Cell> operator +(const tensor<Cell> & a, const Cell & b){
            tensor<Cell> rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val+b;
            return rt;
        }
		friend tensor<Cell> operator -(const tensor<Cell> & a, const Cell & b){
            tensor<Cell> rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val-b;
            return rt;
        }
		friend tensor<Cell> operator *(const tensor<Cell> & a, const Cell & b){
            tensor<Cell> rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val*b;
            return rt;
        }
		friend tensor<Cell> operator /(const tensor<Cell> & a, const Cell & b){
            tensor<Cell> rt;
            rt._num=a._num;
            rt._shp=a._shp;
            rt._val=a._val/b;
            return rt;
        }
		friend tensor<Cell> operator +(const Cell & a, const tensor<Cell> & b){
            tensor<Cell> rt;
            rt._num=b._num;
            rt._shp=b._shp;
            rt._val=a+b._val;
            return rt;
        }
		friend tensor<Cell> operator -(const Cell & a, const tensor<Cell> & b){
            tensor<Cell> rt;
            rt._num=b._num;
            rt._shp=b._shp;
            rt._val=a-b._val;
            return rt;
        }
		friend tensor<Cell> operator *(const Cell & a, const tensor<Cell> & b){
            tensor<Cell> rt;
            rt._num=b._num;
            rt._shp=b._shp;
            rt._val=a*b._val;
            return rt;
        }
		friend tensor<Cell> operator /(const Cell & a, const tensor<Cell> & b){
            tensor<Cell> rt;
            rt._num=b._num;
            rt._shp=b._shp;
            rt._val=a/b._val;
            return rt;
        }
		friend tensor<Cell> transpose(const tensor<Cell> &a,const std::valarray<size_t> index)
		{
			std::valarray<size_t> shp(a.shape().size()), idxs,idxs2(a.shape().size());
			std::valarray<Cell>   val(a.size());
			for (size_t i = 0; i < shp.size(); i++)shp[i] = a.shape()[i < index.size() ? index[i] : i];
			for (size_t i = 0; i < a.size(); i++)
			{
				idxs = tensor<Cell>::loc2idxs(i, a.shape(), a.size());
				for (size_t j = 0; j < shp.size(); j++)idxs2[j] = idxs[j < index.size() ? index[j] : j];
				val[tensor<Cell>::idxs2loc(idxs2, shp)] = a._val[i];
			}
			return tensor<Cell>(shp,val);
		}
        friend tensor<Cell> mul(const tensor<Cell> &a,const tensor<Cell>&b,
                                const std::vector<size_t>&ca,
                                const std::vector<size_t>&cb,
								const Cell(*_inner)(const tensor<Cell>&)
                                ){
            tensor<Cell> rt;
            return rt;
        }
        friend tensor<Cell> cov(const tensor<Cell> &a,const tensor<Cell>&b,
                                const Cell(*_inner)(const tensor<Cell>&,const tensor<Cell>&)
                                ){
            tensor<Cell> rt;
            return rt;
        }
        friend tensor<Cell> solve(const tensor<Cell> &a,const tensor<Cell>&b,
                                const std::vector<size_t>&ca,
                                const std::vector<size_t>&cb,
                                const Cell(*_inner)(const tensor<Cell>&)
                                ){
            tensor<Cell> rt;
            return rt;
        }
        friend tensor<Cell> poll(const tensor<Cell> &a,
                                  const std::vector<size_t>&ca,
                                  const Cell(*_inner)(const tensor<Cell>&)
                                  ){
            tensor<Cell> rt;
            return rt;
        }
#include "tensor_tnna.hpp"
    };
}
#endif /* tensor_h */
