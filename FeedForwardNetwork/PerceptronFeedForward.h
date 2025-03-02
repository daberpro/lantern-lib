#pragma once
#include "../pch.h"
#include "Perceptron.h"
#include <Vector.h>

namespace latern {

    namespace perceptron {
        void PerceptronFeedForward(Perceptron* objective, utility::Vector<Perceptron*>& fix_position_node);
        void PerceptronFeedForward(utility::Vector<Perceptron*>& fix_position_node);
        void PerceptronUpdateCalculation(utility::Vector<Perceptron*>& fix_position_node);
    }

}