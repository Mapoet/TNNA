//
//  object_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef object_tnna_h
#define object_tnna_h
namespace TNNA{
    template<int n, typename Scale>
    struct object{
        public:
        
    };
    template<int n, typename Scale>
	struct point :public object<n,Scale>,public tensor<Scale>{
		point(const Scale&s=Scale(1.0)) :tensor<Scale>({ n }, s, 1){}
    };
    
    
}

#endif /* object_tnna_h */
