#pragma once
#include "Node.h"
#include <Vector.h>
/**
 * FLOATING_TYPE is a macro which define on Node.h file 
 */

namespace lantern {
    /**
     * @brief Reverse Mode Auto Gradient
     * 
     * Example usage:
     * 
     * ```cpp
     * lantern::Node n2(1.0f);
     * lantern::Node n3 = n1 + n2;
     * lantern::ReverseModeAD(n3);
     * ```
     * 
     */
    void ReverseModeAD(Node&);
}

