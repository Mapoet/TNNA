//
//  iostream_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef iostream_tnna_h
#define iostream_tnna_h
#include <ctime>
#include <chrono>
#include <vector>
namespace TNNA{
template<typename Flow>
    struct iostream{
    protected:
        tensor<Flow> _datas;
        std::chrono::steady_clock::time_point _pt;
		std::chrono::milliseconds _during;
        std::timed_mutex          _mutex;
		iostream(const tensor<Flow>&datas = tensor<Flow>()) :_datas(datas), _during(std::chrono::milliseconds(200)){ _pt = std::chrono::steady_clock::now(); }
    public:
		virtual tensor<Flow> fresh() = 0;
		virtual bool    during() = 0;
		virtual void    update(const tensor<Flow>&datas, const std::chrono::milliseconds&during = std::numeric_limits<std::chrono::milliseconds>::max()) = 0;
    };
    
    template<typename Flow>
    struct DataStream:public iostream<Flow>{
    protected:
		DataStream(const tensor<Flow>&data = tensor<Flow>()) :iostream<Flow>(data){}
    public:
		static std::shared_ptr<iostream<Flow>>New(const tensor<Flow>&data = tensor<Flow>())
        {
			return std::shared_ptr<iostream<Flow>>(new DataStream<Flow>(data));
        }
		virtual tensor<Flow>     fresh(){
            std::lock_guard<std::timed_mutex> guard(this->_mutex);
            return this->_datas;
        }
        virtual bool  during(){
            std::lock_guard<std::timed_mutex> guard(this->_mutex);
			return this->_during.count() - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - this->_pt).count() > 0;
		}
		virtual void    update(const tensor<Flow>&datas, const std::chrono::milliseconds&during = std::numeric_limits<std::chrono::milliseconds>::max()){
			assert (datas.shape().size() <= 1||(datas.shape().size()==2&&datas.shape()[0]==1));
            std::lock_guard<std::timed_mutex> guard(this->_mutex);
            this->_datas = datas;
			this->_datas.reshape({ this->_datas.shape().max() });
            this->_during = during;
            this->_pt = std::chrono::steady_clock::now();
        }
    };
}
#endif /* datastream_tnna_h */
