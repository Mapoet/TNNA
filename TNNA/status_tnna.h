//
//  status_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef status_tnna_h
#define status_tnna_h
namespace TNNA{
    enum cellStatusType{
        cellStatus_Null = 0x0000,
        cellStatus_Dead = 0x0000,
        cellStatus_Alived = 0x0001,
        cellStatus_Pause = 0x0002,
        cellStatus_Resume = 0x0004,
        cellStatus_Learning = 0x0010,
        cellStatus_Thinking = 0x0100,
        
        cellStatus_Feedback = 0x0001,
        cellStatus_Transmit = 0x0002,
        cellStatus_IStr = 0x0004,
        cellStatus_OStr = 0x0008
	};
	enum cellStreamType{
		cellStream_Hidden = 0x000,
		cellStream_Input = 0x001,
		cellStream_Output = 0x002
	};
    enum cellActiveType{
        cellActive_ReLU,
        cellActive_WeakLU,
        cellActive_Sigmoid,
        cellActive_Linear,
		cellActive_Atanh,
		cellActive_Functional
    };
    enum cellTransmitType{
        cellTransmit_Linear,
        cellTransmit_Simgion,
		cellTransmit_Atan,
		cellTransmit_Functional
    };
    enum cellGraphLayerType{
        cellGraphLayer_MarkovChain,
        cellGraphLayer_DeepBeliefNetWork,
        cellGraphLayer_DeepResidualNetWork,
        cellGraphLayer_DeepDenseNetWork
    };
    class cellStatus{
        cellStatusType _status;
    public:
        cellStatus(const cellStatusType&status = cellStatus_Alived) :_status(status){}
        void SetStatus(const cellStatusType&status){
            _status = status;
        }
        void Alived(const bool &isenable){
            _status = (cellStatusType)((size_t)_status + (isenable?(size_t)cellStatus_Alived:0) - (size_t)_status % 16);
        }
        bool Alived(){ return ((size_t)_status) % 16 == cellStatus_Alived; }
        void Dead(const bool &isenable){
            _status = (cellStatusType)((size_t)_status + (isenable ?  (size_t)cellStatus_Dead:0) - (size_t)_status % 16);
        }
        bool Dead(){ return ((size_t)_status) % 16 == cellStatus_Dead; }
        void Pause(const bool &isenable){
            _status = (cellStatusType)((size_t)_status + (isenable ?  (size_t)cellStatus_Pause:0) - (size_t)_status % 16);
        }
        bool Pause(){ return ((size_t)_status) % 16 == cellStatus_Pause; }
        void Resume(const bool &isenable){
            _status = (cellStatusType)((size_t)_status + (isenable ? (size_t)cellStatus_Resume:0) - (size_t)_status % 16);
        }
        bool Resume(){ return ((size_t)_status) % 16 == cellStatus_Resume; }
        
        void Learning(const cellStatusType &isenable){
            size_t bit = (_status / 16) % 16;
            _status = (cellStatusType)((size_t)_status + ((size_t)isenable - bit) * 16);
        }
        cellStatusType Learning(){ return (cellStatusType)((_status / 16) % 16); }
        void Thinking(const cellStatusType &isenable){
            size_t bit = (_status / (16*16)) % 16;
            _status = (cellStatusType)((size_t)_status + ((size_t)isenable - bit) * (16*16));
        }
        cellStatusType Thinking(){ return (cellStatusType)((_status / (16 * 16)) % 16); }
        void DeepLearning(const cellStatusType &isenable){
            size_t bit = (_status / (16 * 16 * 16)) % 16;
            _status = (cellStatusType)((size_t)_status + ((size_t)isenable - bit) * (16 * 16 * 16));
        }
        cellStatusType DeepLearning(){ return (cellStatusType)((_status / (16 * 16 * 16)) % 16); }
        void DeepThinking(const cellStatusType &isenable){
            size_t bit = (_status / (16 * 16 * 16 * 16)) % 16;
            _status = (cellStatusType)((size_t)_status + ((size_t)isenable - bit) * (16 * 16 * 16 * 16));
        }
        cellStatusType DeepThinking(){ return (cellStatusType)((_status / (16 * 16 * 16 * 16)) % 16); }
    };
    
}

#endif /* status_tnna_h */
