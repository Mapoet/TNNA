//
//  field_tnna.h
//  TNNA
//
//  Created by Mapoet Niphy on 2018/11/3.
//  Copyright © 2018年 Mapoet Niphy. All rights reserved.
//

#ifndef field_tnna_h
#define field_tnna_h
#include "object_tnna.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_odeiv2.h>
namespace TNNA{
    /*
    Runge-Kutta
        0  |
        c_2|a_21
        c_3|a_31 a_32
        :  |:    :
        c_s|a_s1 a_s2 ... a_s,s-1
        ------------------------------
           |b_1  b_2  ... b_s-1    b_s   
        y_n+1=y_n+h\sum_{i=1}^s b_i k_i
        k_i= f(t_n+c_i h,y_n+h\sum_{j=2}^{i-1}a_ij k_j )
    */
    template<int n, typename Scale>
    struct field:public object<n,Scale>{
        std::valarray<std::valarray<Scale> > _axis;
        tensor<Scale> _data;
        field(const  std::valarray<std::valarray<Scale> >&axis):_axis(axis){
             std::valarray<size_t> dim(_axis.size());
             for(size_t i=0;i<dim.size();i++)dim[i]=_axis[i].size();
            _data.resize(dim);
            for(size_t ix=0;ix<_axis[0].size();ix++)
            for(size_t iy=0;iy<_axis[1].size();iy++)
            for(size_t iz=0;iz<_axis[2].size();iz++){
                _data[{ix,iy,iz}]=ix*iy*iz;
            }
        }
        tensor<Scale> grad(size_t id=0){
            tensor<Scale> r=_data;
            for(size_t ix=0;ix<_axis[0].size();ix++)
            for(size_t iy=0;iy<_axis[1].size();iy++)
            for(size_t iz=0;iz<_axis[2].size();iz++)
            switch (id)
            {
                case 1:if(ix==0)r[{ix,iy,iz}]= (_data[{ix+1,iy,iz}]-_data[{ix,iy,iz}])/(_axis[0][ix+1]-_axis[0][ix]);
                        else if(ix==_axis[0].size()-1)r[{ix,iy,iz}]= (_data[{ix,iy,iz}]-_data[{ix-1,iy,iz}])/(_axis[0][ix]-_axis[0][ix-1]);
                        else r[{ix,iy,iz}]= (_data[{ix+1,iy,iz}]-_data[{ix-1,iy,iz}])/(_axis[0][ix+1]-_axis[0][ix-1]);
                    break;
                case 2:if(ix==0)r[{ix,iy,iz}]= (_data[{ix,iy+1,iz}]-_data[{ix,iy,iz}])/(_axis[1][iy+1]-_axis[1][iy]);
                        else if(ix==_axis[1].size()-1)r[{ix,iy,iz}]= (_data[{ix,iy,iz}]-_data[{ix,iy-1,iz}])/(_axis[1][iy]-_axis[1][iy-1]);
                        else r[{ix,iy,iz}]= (_data[{ix,iy+1,iz}]-_data[{ix,iy-1,iz}])/(_axis[1][iy+1]-_axis[1][iy-1]);
                    break;
                case 3:if(ix==0)r[{ix,iy,iz}]= (_data[{ix,iy,iz+1}]-_data[{ix,iy,iz}])/(_axis[2][iz+1]-_axis[2][iz]);
                        else if(ix==_axis[2].size()-1)r[{ix,iy,iz}]= (_data[{ix,iy,iz}]-_data[{ix,iy,iz-1}])/(_axis[2][iz]-_axis[2][iz-1]);
                        else r[{ix,iy,iz}]= (_data[{ix,iy,iz+1}]-_data[{ix,iy,iz-1}])/(_axis[2][iz+1]-_axis[2][iz-1]);
                    break;
                default:
                    break;
            }

        }
    };
}
#endif /* field_tnna.h */
