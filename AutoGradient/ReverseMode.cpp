#include "ReverseMode.h"

namespace latern{

    // !check for the parent pointer
    // if the pointer same as the current node pointer
    // just throw an error

    void ReverseModeAD(Node& first_node){
        
        if(first_node.parents == 0){
            return;
        }
        
        CalculateGradientO(first_node);
        utility::Vector<Node*> all_parents;

        if(first_node.parents[1] != nullptr){
            all_parents = {
                first_node.parents[0],
                first_node.parents[1]
            };
        }else{
            all_parents = {
                first_node.parents[0]
            };
        }

        Node* current_node = nullptr;
        while(!all_parents.empty()){
            current_node = all_parents.back();
            CalculateGradient(*(current_node));
            all_parents.pop_back();
            
            if(current_node->parents[0] != nullptr ){
                all_parents.push_back(current_node->parents[0]);
            }
            if(first_node.parents[1] != nullptr){
                if(!(current_node->parents[1] == nullptr )){
                    all_parents.push_back(current_node->parents[1]);
                }
            }
        }
    }
}
