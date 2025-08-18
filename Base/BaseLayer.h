#pragma once
#include "../pch.h"
#include "../Headers/Vector.h"

/*
* =================================================================================
* Lantern Base layer
* in here we define a base layer class for all layer 
* this used to handle multiply layer
* =================================================================================
*/

namespace lantern {

    class BaseLayer {
    protected:

        lantern::utility::Vector<uint32_t> m_LayersSize;

    public:

        BaseLayer() = default;

        lantern::utility::Vector<uint32_t>* GetAllLayerSizes() {
            return &this->m_LayersSize;
        }

        virtual void PrintLayerInfo() {}
        virtual ~BaseLayer() {}

    };

}