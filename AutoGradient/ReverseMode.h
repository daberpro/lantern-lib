#pragma once
#include "Node.h"
#include <Vector.h>
/**
 * FLOATING_TYPE is a macro which define on Node.h file 
 */

namespace latern {
    /**
     * @brief Reverse Mode Auto Gradient
     * 
     * Example usage:
     * 
     * ```cpp
     * latern::Node n2(1.0f);
     * latern::Node n3 = n1 + n2;
     * latern::ReverseModeAD(n3);
     * ```
     * 
     */
    void ReverseModeAD(Node&);
}

