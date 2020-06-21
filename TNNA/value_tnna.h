//
//  value_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef value_tnna_h
#define value_tnna_h
namespace TNNA{

    template<typename Data>
    class value{
		template<typename Scales,typename Flows,typename Datas>
		friend class cell;
	protected:
		Data _data;
		value(const Data&data = Data()) :_data(data){}
		virtual void print(std::ostream&ios) = 0;
		public:
		Data& data(){return _data;}
    };
	template<typename Data>
	class DataValue :public value<Data>{
	protected:
		DataValue(const Data&data=Data()) :value<Data>(data){}
		virtual void print(std::ostream&ios){
			ios <<"Value:"<<this->_data;
		}
	public:
		static std::shared_ptr<value<Data>>New(Data data = Data()){
			return std::shared_ptr<value<Data>>(new DataValue<Data>(data));
		}
	};
    
}

#endif /* value_h */
