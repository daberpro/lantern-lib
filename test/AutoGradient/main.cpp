#include <iostream>
#include "../../AutoGradient/ReverseMode.h"
#include <gtest/gtest.h>
#include <cmath>

FLOATING_TYPE Sigmoid(FLOATING_TYPE& x){
    return (1/(1+exp(-x)));
}

/**
 * Sg -> Sigmoid
 * Id -> Identifier
 * Mult -> Multiply
 * Add -> Addition
 * Div -> Division
 * Subs -> Subtraction
 * Ln -> NaturalLog
 * Ex -> Exp
 * Sin -> Sin
 * Cos -> Cos
 * Tab -> Tan
 */

TEST(AutoGradeitn,SgSgId) {
    lantern::Node a(2.5);
    lantern::Node sg1 = lantern::Sigmoid(a); 
    lantern::Node sg2 = lantern::Sigmoid(sg1);
    lantern::ReverseModeAD(sg2);
    
    ASSERT_EQ(sg2.value,Sigmoid(sg1.value));
    ASSERT_EQ(sg1.value,Sigmoid(a.value));
    ASSERT_EQ(sg2.gradient,sg2.value * (1 - sg2.value));
    ASSERT_EQ(sg1.gradient,(sg2.value * (1 - sg2.value)) * (sg1.value * (1 - sg1.value)));
    ASSERT_EQ(a.gradient,1.0);
}

int main(){
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}