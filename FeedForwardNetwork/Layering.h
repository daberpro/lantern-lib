#pragma once
#include "Perceptron.h"

namespace lantern {

    namespace perceptron {

        template <uint32_t level = 0,typename... Args>
        void SetLayer(Args&... q){
            ((q.layer = level),...);
        }

    }

}