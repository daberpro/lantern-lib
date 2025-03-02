#pragma once
#include "../pch.h"
#include <Node.h>
#include <Vector.h>

namespace latern {

    /**
     * @brief FeedForward the current node root and bind fix position node into the second argument
     * 
     * Example usage:
     * 
     * ```cpp
     * latern::Node n1(1.0f);
     * latern::Node n2(1.0f);
     * latern::Node n3 = n1 + n2;
     * latern::utility::Vector<Node*> fix_position_node;
     * latern::FeedForward(n3,fix_position_node);
     * ```
     *
     * @param objective 
     * @param fix_position_node 
     */
    void FeedForward(Node* objective, utility::Vector<Node*>& fix_position_node);

    /**
     * @brief FeedForward only for update calculation value for each node
     * 
     * Example usage:
     * 
     * ```cpp
     * latern::Node n2(1.0f);
     * latern::Node n3 = n1 + n2;
     * latern::utility::Vector<Node*> fix_position_node;
     * latern::FeedForward(fix_position_node);
     * ```
     * 
     * @param fix_position_node 
     */
    void FeedForward(utility::Vector<Node*>& fix_position_node);

    /**
     * @brief update calculation for single node
     * 
     * Example usage:
     * 
     * @code {.cpp}
     * latern::Node n2(1.0f);
     * latern::Node n3 = n1 + n2;
     * n1.value = 2.0;
     * n2.value = 4.0;
     * latern::UpdateCalculation(n3);
     * @endcode
     * 
     * @param fix_position_node 
     */
    void UpdateCalculation(utility::Vector<Node*>& fix_position_node);

}