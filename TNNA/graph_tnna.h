//
//  graph_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef graph_tnna_h
#define graph_tnna_h
#include <tuple>
#include <chrono>
#include "cell_tnna.h"
#include "iostream_tnna.h"
namespace TNNA{
    template<typename Scale,typename Flow,typename Data>
    class graph{
	public:
		friend class cell<Scale, Flow, Data>;
		typedef std::map<size_t, size_t> slice;
		typedef cell<Scale, Flow, Data>				  Node;
		typedef std::shared_ptr<iostream<Flow>>       IOStream;
		typedef std::shared_ptr<transmit<Scale, Flow>> Transmit;
		typedef std::shared_ptr<active<Scale, Flow>>   Active;
		typedef std::shared_ptr<value<Data>>          Value;
		typedef std::vector<std::tuple<Value, Active>> Nodes;
		typedef std::vector<std::tuple<size_t, size_t,Transmit>> Links;
        typedef std::vector<std::tuple<cellStreamType, size_t, IOStream>> LabelIOStream;
        typedef std::shared_ptr<graph<Scale, Flow, Data> > GRAPH;
      private:
		size_t										_nbat;
		std::vector<typename Node::Node>			_nodes;
		std::map<Node*, IOStream >					_istrs,_ostrs;
        std::chrono::milliseconds                   _msleep;
        void  LearningStatus(const cellStatusType &istatus){
            for (auto &it : _nodes)
                if (it != nullptr)it->LearningStatus(istatus);
        }
        void  ThinkingStatus(const cellStatusType &istatus){
            for (auto &it : _nodes)
                if (it != nullptr)it->ThinkingStatus(istatus);
        }
        void  SetIstr(const tensor<Flow>& is, const std::chrono::milliseconds& t = std::numeric_limits<std::chrono::milliseconds>::max()){
            assert(is.shape().size() == 2 && _istrs.size() == is.shape()[0]);
            size_t i = 0;
            for (auto it = _istrs.begin(); it != _istrs.end(); it++)
                it->second->update(is(slice({ { 0, i++ } })), t);
        }
        void  SetOstr(const tensor<Flow>& os, const std::chrono::milliseconds& t = std::numeric_limits<std::chrono::milliseconds>::max()){
            assert(os.shape().size() == 2 && _ostrs.size() == os.shape()[0]);
            size_t i = 0;
            for (auto it = _ostrs.begin(); it != _ostrs.end(); it++)
                it->second->update(os(slice({ { 0, i++ } })), t);
        }
        void  GetOstr(tensor<Flow>& os){
            size_t i = 0;
            for (auto it = _ostrs.begin(); it != _ostrs.end(); it++)
            {
                if (i == 0)
                    os.resize({ _ostrs.size(), it->second->fresh().data().size() });
                os(slice({ { 0, i++ } }), it->second->fresh().data());
            }
        }	
        void discell(typename Node::Self* self){		
           for (auto it : self->_istr){
				it.first->_ostr.erase(self);
			}
			for (auto it :self-> _ostr){
				it.first->_ostr.erase(self);
			}
			{
			    auto loc = _istrs.find(self);
			    if (loc != _istrs.end())_istrs.erase(loc);
			}
			{
			    auto loc = _ostrs.find(self);
			    if (loc != _ostrs.end())_ostrs.erase(loc);
			}
			{
			    auto loc = std::find_if(_nodes.begin(),_nodes.end(),[&](const typename Node::Node&node){return self==node.get();});
			    if (loc != _nodes.end())_nodes.erase(loc);
			}
		}
        graph() : _msleep(100), _nbat(1) {}
      public:
      static typename Node::Node Generate(graph *root, const Value &value, const Active &active)
      {
            auto node = Node::New(root, value, active);
            root->_nodes.emplace_back(node);
            return node;
      }
      static typename Node::Node Generate(graph *root, const Data &data, const typename FunctionalActive<Scale, Flow>::Kernel &kernel,const std::valarray<Flow>&args)
      {
          auto node=Node::New(root, DataValue<Data>::New(data), FunctionalActive<Scale,Flow>::New(Scale(0.1),kernel,args));
          root->_nodes.emplace_back(node);
          return node;
      }
      static GRAPH New(){
          return GRAPH(new graph<Scale,Flow,Data>());
      }
      ~graph()
      {
          _nodes.clear();
          _istrs.clear();
          _ostrs.clear();
		}
		void print(std::ostream&ios){
			ios << "graph:{nbat:"<<_nbat<<",cells:{\n";
			for (auto it : _nodes)
				it->print(ios);
			ios << "}.\n";
		}
		void  BuildStruct(const Nodes &cells, const Links&links, const LabelIOStream&streams){
			for (auto it : cells)
				_nodes.emplace_back(Node::New(this, std::get<0>(it), std::get<1>(it)));
			for (auto it : links)
				_nodes[std::get<0>(it)]->insertO(_nodes[std::get<1>(it)], std::get<2>(it));
			for (auto it : streams)
				switch (std::get<0>(it))
			{
				case cellStream_Input:	_istrs.emplace(_nodes[std::get<1>(it)].get(), std::get<2>(it)); break;
				case cellStream_Output:	_ostrs.emplace(_nodes[std::get<1>(it)].get(), std::get<2>(it)); break;
				default: break;
			}
		}
		void  StartCell(const size_t nbat = 1, const std::chrono::milliseconds&msleep = std::chrono::milliseconds(100)){
			_nbat = nbat;
            _msleep=msleep;
			for (auto &it : _nodes)
				if(it!=nullptr)it->StartCell(msleep);
		}
		void  ReSetBat(const size_t nbat = 1, const std::chrono::milliseconds&msleep = std::chrono::milliseconds(100)){
			for (size_t i = 0; i < _nodes.size(); i++)
				std::get<0>(_nodes[i]->_living).Pause();
            for (size_t i = 0; i < _nodes.size(); i++)
                _nodes[i]->clear();
			_nbat = nbat;
            _msleep=msleep;
			for (size_t i = 0; i < _nodes.size(); i++)
                std::get<0>(_nodes[i]->_living).Resume();
		}
		void  Learning(const tensor<Flow>& is, const tensor<Flow>& os,const bool&related=false, const std::chrono::milliseconds& t = std::numeric_limits<std::chrono::milliseconds>::max()){
			SetIstr(is, t);
			SetOstr(os, t);
            LearningStatus(cellStatus_Alived);
            std::this_thread::sleep_for(t);
            LearningStatus(cellStatus_Dead);
            if(related)return;
            for (size_t i = 0; i < _nodes.size(); i++)
            {
                _msleep=_msleep>_nodes[i]->_msleep?_msleep:_nodes[i]->_msleep;
                std::get<0>(_nodes[i]->_living).Pause();
            }
            std::this_thread::sleep_for(_msleep);
            for (size_t i = 0; i < _nodes.size(); i++)
                _nodes[i]->clear();
            for (size_t i = 0; i < _nodes.size(); i++)
                std::get<0>(_nodes[i]->_living).Resume();
		}
		void  Thinking(const tensor<Flow>& is, tensor<Flow>& os,const bool&related=false, const std::chrono::milliseconds& t = std::numeric_limits<std::chrono::milliseconds>::max()){
			SetIstr(is, t);
            ThinkingStatus(cellStatus_Alived);
            std::this_thread::sleep_for(t);
            ThinkingStatus(cellStatus_Dead);
			GetOstr(os);
            if(related)return;
            for (size_t i = 0; i < _nodes.size(); i++)
            {
                _msleep=_msleep>_nodes[i]->_msleep?_msleep:_nodes[i]->_msleep;
                std::get<0>(_nodes[i]->_living).Pause();
            }
            std::this_thread::sleep_for(_msleep);
            for (size_t i = 0; i < _nodes.size(); i++)
                _nodes[i]->clear();
            for (size_t i = 0; i < _nodes.size(); i++)
                std::get<0>(_nodes[i]->_living).Resume();
		}
		typename Node::Node Get(const size_t& i){ assert(i<_nodes.size());return _nodes[i]; }
        typename Node::Node Remove(const size_t&i){
            assert(i<_nodes.size());
            auto node=_nodes[i];
            if(node!=nullptr)this->discell(node.get());
            return node;
        }
        void Remove(typename Node::Node&node){
            if(node!=nullptr&&node->_root==this)this->discell(node.get());
        }
	};
}

#endif /* graph_tnna_h */
