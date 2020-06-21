//
//  cell_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef cell_tnna_h
#define cell_tnna_h
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <tuple>
#include <memory>
#include "status_tnna.h"
#include "value_tnna.h"
#include "active_tnna.h"
#include "transmit_tnna.h"
#include "iostream_tnna.h"
#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
namespace TNNA{
		template<typename Scale, typename Flow, typename Data>
    class graph;
    template<typename Scale,typename Flow,typename Data>
	class cell{
		template<typename Scales, typename Flows, typename Datas>
		friend class graph;
		typedef std::map<size_t, size_t> slice;
		typedef std::valarray<size_t>    idxs;
		typedef std::shared_ptr<iostream<Flow>>       IOStream;
        typedef std::shared_ptr<transmit<Scale,Flow>> Transmit;
        typedef std::shared_ptr<active<Scale,Flow>>   Active;
		typedef std::shared_ptr<value<Data>>          Value;
		typedef graph<Scale, Flow, Data>              Root;
        typedef cell<Scale,Flow,Data>                 Self;
		typedef std::shared_ptr<Self>                 Node;
		size_t									   _id;
		const Root*								   _root;
		Active                                     _active;
		Value                                      _value;
		std::map<Self*, Transmit >				   _istr, _ostr;
		mutable std::tuple<cellStatus, std::thread, std::timed_mutex> _living;
		std::chrono::milliseconds					_msleep;
		size_t generateid(){
			static size_t id = 0;
			return id++;
		}
		cell(const Root* root = nullptr, const Value&value = Value(), const Active&active = Active()) :_id(generateid()), _root(root), _value(value), _active(active){}
		template<typename Vs, typename As>
		cell(const Root* root, const Vs*value, const As*active) : _id(generateid()), _root(root), _value(value), _active(active){}
		bool update_istr(const bool&feedback){
			size_t i = 0, itype = _istr.size();
			tensor<Flow> rt,use;
			auto loc = _root->_istrs.find(this);
			if (loc != _root->_istrs.end() &&
				loc->second->during())
			{
				std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
				_active->in() = loc->second->fresh();
			}
			else
			{
				for (auto it = _istr.begin(); it != _istr.end(); it++)
				if (it->first->isout() == false)return false;
				rt.resize({ itype==0?1:itype, _root->_nbat });	
				i = 0;
				for (auto it = _istr.begin(); it != _istr.end(); it++)
				{
					if (feedback&&isin())
					{
						use = it->first->out(this);
						if (use.size() == 0)return false;
						it->first->out(this, it->second->update(in(it->first), use));
					}
					use = it->first->out(this);
					if (use.size() == 0)return false;
					rt(slice({ { 0, i } }), it->second->predict(use).data());
					i++;
				}
				if (loc != _root->_istrs.end() &&
					(!loc->second->during()))
					loc->second->update(rt, _msleep * 2);
				std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
				_active->in() = rt;
			}
			return true;
		}
		bool update_ostr(const bool&feedback){
			size_t itype = _istr.size(), otype = _ostr.size();
			if (isin() == false)return false;
			auto loc = _root->_ostrs.find(this);
			if (loc != _root->_ostrs.end())
			{
				if (loc->second->during())
				{
					std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
					_active->out() = loc->second->fresh();
					if (feedback){
						auto err = _active->feedback(itype == 0 ? 1 : itype, otype == 0 ? 1 : otype, _root->_nbat);
						//std::cout << "Learning id:" << _id << " with err:" << err << "!\n";
					}
				}
				else
				{
					if (isout() == false)
					{
						std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
						_active->act(itype == 0 ? 1 : itype, otype == 0 ? 1 : otype, _root->_nbat);
					}
					std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
					loc->second->update(_active->out(), _msleep);
				}
			}
			else
			{
				if (isout() == false)
				{
					std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
					_active->act(itype == 0 ? 1 : itype, otype == 0 ? 1 : otype, _root->_nbat);
				}
				else{
					std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
					if (feedback){
						auto err = _active->feedback(itype == 0 ? 1 : itype, otype == 0 ? 1 : otype, _root->_nbat);
						//std::cout << "Learning id:" << _id << " with err:" << err << "!\n";
					}
					else
						_active->act(itype == 0 ? 1 : itype, otype == 0 ? 1 : otype, _root->_nbat);
				}
			}
			return true;
		}
		tensor<Flow> in(const Self*self)const{
			std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
			if (self == nullptr)
				return _active->in();
			else{
				size_t i = std::distance(_istr.begin(), _istr.find(const_cast<Self*>(self)));
				auto temp = _active->in();
				return (i >= _istr.size() || _active->in().size() == 0) ? tensor<Flow>({ _root->_nbat }, std::valarray<Flow>()) : temp(slice({ { 0, i } }));
			}
		}
		tensor<Flow> out(const Self*self)const{
			std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
			if (self == nullptr)
				return _active->out();
			else{
				size_t i = std::distance(_ostr.begin(), _ostr.find(const_cast<Self*>(self)));
				auto temp = _active->out();
				return  (i >= _ostr.size() || _active->out().size() == 0) ? tensor<Flow>({ _root->_nbat }, std::valarray<Flow>()) : temp(slice({ { 0, i } }));
			}
		}
		bool isin()const{
			std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
			bool flag = true;
			for (size_t i = 0; i < _active->in().size(); i++)
			if (_active->in().data()[i].size() == 0)flag = false;
			flag = (flag&(_active->in().size() != 0));
			return flag;
		}
		bool isout()const{
			std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
			bool flag = true;
			for (size_t i = 0; i < _active->out().size(); i++)
			if (_active->out().data()[i].size() == 0)flag = false;
			flag = (flag&(_active->out().size() != 0));
			return flag;
		}
		void in(const Self*self, const tensor<Flow>&delta){
			std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
			size_t i = std::distance(_istr.begin(), _istr.find(const_cast<Self*>(self)));
			if (i >= _istr.size())return;
			for (size_t j = 0; j < _root->_nbat; j++)
				_active->in()[{ i, j }] = _active->in()[{ i, j }] + delta[{ j }];
		}
		void out(const Self*self, const tensor<Flow>&delta){
			std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
			size_t i = std::distance(_ostr.begin(), _ostr.find(const_cast<Self*>(self)));
			if (i >= _ostr.size())return;
			for (size_t j = 0; j < _root->_nbat; j++)
				_active->out()[{ i, j }] = _active->out()[{ i, j }] + delta[{ j }];
		}
        void clear(){
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            _active->clear();
        }
    public:
        static Node
			New( const Root* root = nullptr, const Value&value = Value(), const Active&active = Active()){
				return Node(new Self(root, value, active));
			}
        template<typename Vs,typename As>
		static std::shared_ptr<cell<Scale, Flow, Data>>
			New(const Root* root, const Vs*value, const As*active){
			return std::shared_ptr<cell<Scale, Flow, Data>>(new cell<Scale, Flow, Data>(root,value, active));
		}
		~cell(){
			std::get<0>(_living).Alived(false);
			if (std::get<1>(_living).joinable())
				std::get<1>(_living).join();
		}
		Data &data(){
			return _value->data();
		}
		operator Value()const{
			return _value;
		}
		void print(std::ostream&ios){
			ios << "cell:{id:"<<_id<<",";
			_value->print(ios);
			ios << ",\n";
			_active->print(ios);
			ios << ",\nOutter:{\n";
			for (auto it : _ostr){
				ios << "\t(id:" << it.first->_id << ",";
				it.second->print(ios);
				ios<< ")\n";
			}
			ios << "\t}\n}\n";
		}
        Self* next(const std::string&path){return nullptr;}
        Self* back(const std::string&path){return nullptr;}
        template<typename Ts>
        bool insertI(Self*cell,const Ts* ts){
            if(cell==nullptr)return false;
			this->removeO(cell);
            std::shared_ptr<Ts> t(ts);
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            cell->_ostr.emplace(this,t);
            _istr.emplace(cell,t);
            _active->in()=tensor<Flow>();
            cell->_active->out()=tensor<Flow>();
            return true;
        }
        template<typename Ts>
        bool insertO(Self*cell,const Ts* ts){
            if(cell==nullptr)return false;
			this->removeI(cell);
            std::shared_ptr<Ts> t(ts);
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            cell->_istr.emplace(this,t);
            _ostr.emplace(cell,t);
            _active->out()=tensor<Flow>();
            cell->_active->in()=tensor<Flow>();
            return true;
        }
        void removeI(Self*cell){
            if(cell==nullptr)return;
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            cell->_ostr.erase(this);
            _istr.erase(cell);
            _active->in()=tensor<Flow>();
            cell->_active->out()=tensor<Flow>();
        }
        void removeO(Self*cell){
            if(cell==nullptr)return;
            cell->_istr.erase(this);
            _ostr.erase(cell);
            _active->out()=tensor<Flow>();
            cell->_active->in()=tensor<Flow>();
        }
        bool insertI(Node&cell,const Transmit& t){
            if(cell==nullptr)return false;
			this->removeO(cell.get());
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            cell->_ostr.emplace(this,t);
            _istr.emplace(cell.get(),t);
            _active->in()=tensor<Flow>();
            cell->_active->out()=tensor<Flow>();
            return true;
        }
        bool insertO(Node&cell,const Transmit& t){
            if(cell==nullptr)return false;
			this->removeI(cell.get());
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            cell->_istr.emplace(this,t);
            _ostr.emplace(cell.get(),t);
            _active->out()=tensor<Flow>();
            cell->_active->in()=tensor<Flow>();
            return true;
        }
        void removeI(Node&cell){
            if(cell==nullptr)return;
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            cell->_ostr.erase(this);
            _istr.erase(cell.get());
            _active->in()=tensor<Flow>();
            cell->_active->out()=tensor<Flow>();
        }
        void removeO(Node&cell){
            if(cell==nullptr)return;
            std::lock_guard<std::timed_mutex> guard(std::get<2>(_living));
            cell->_istr.erase(this);
            _ostr.erase(cell.get());
            _active->out()=tensor<Flow>();
            cell->_active->in()=tensor<Flow>();
        }
		void Learning(){if(update_istr(true))update_ostr(true);}
		void Thinking(){if(update_istr(false))update_ostr(false);}
		void  LearningStatus(const cellStatusType &istatus){
			std::get<0>(_living).Learning(istatus);
		}
		void  ThinkingStatus(const cellStatusType &istatus){
			std::get<0>(_living).Thinking(istatus);
		}
		void  StartCell(const std::chrono::milliseconds&msleep = std::chrono::milliseconds(100)
			){
			//size_t itype = _istr.size(), otype = _ostr.size();
			//_active->in().resize({ itype == 0 ? 1 : itype, _root->_nbat });
			//_active->out().resize({ otype == 0 ? 1 : otype, _root->_nbat });
			std::get<0>(_living).Alived(true);
			std::get<1>(_living) = std::thread(CellWork, this, msleep);
		}
		void Pause(){
            std::get<0>(_living).Pause(true);
            std::this_thread::sleep_for(_msleep);
		}
		void Resume(){
            std::get<0>(_living).Resume(true);
            std::this_thread::sleep_for(_msleep);
		}
		bool CellWorking(){
            std::chrono::steady_clock::time_point pt=std::chrono::steady_clock::now();
			if (std::get<0>(_living).Dead()||_active==nullptr)return false;
			if (std::get<0>(_living).Learning() == cellStatus_Alived || std::get<0>(_living).Learning() == cellStatus_Resume)
			{
				this->Learning();
			}
			if (std::get<0>(_living).Thinking() == cellStatus_Alived || std::get<0>(_living).Thinking() == cellStatus_Resume)
			{
				this->Thinking();
			}
            std::chrono::milliseconds dms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pt) ;
            if(dms.count()>_msleep.count())_msleep=2*dms;//std::cout<<dms.count()<<std::endl;
			return true;
		}
		static int CellWork(cell<Scale, Flow, Data> *cell, const std::chrono::milliseconds&msleep
			){
			if (cell == nullptr)return -1;
			cell->_msleep = msleep;
			while (true){
				if (std::get<0>(cell->_living).Dead())
					return 0;
				if (std::get<0>(cell->_living).Pause())goto SLEEP;
				if (cell->CellWorking() != true)
					return 1;
			SLEEP:
				std::this_thread::sleep_for(cell->_msleep);
			}
			return 0;
		}
    };
}
#endif /* cell_h */
