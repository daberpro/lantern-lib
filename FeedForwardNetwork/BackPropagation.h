#pragma once
#include "Perceptron.h"
#include <Vector.h>
#include <Optimizer/GradientDescent.h>
#include <Optimizer/AdaptiveMomentEstimation.h>
#include <Optimizer/RootMeanSquaredPropagation.h>
#include <Optimizer/StochasticGradientDescentWithMomentum.h>
#include <Optimizer/AdaptiveGradientDescent.h>
#include <SymbolDerivative.h>
#include <typeinfo>

static af::array broad_mult(af::array &lhs, af::array &rhs){
	return lhs * rhs;
}

namespace lantern {
    namespace perceptron {

        #ifdef OPTIMIZE_VERSION
        /**
         * @brief Calculate gradient for optimize
         * 
         * @tparam Optimizer 
         * @param objective 
         * @param opt 
         */
        template <typename Optimizer>
        void CalculateGradient(Perceptron &objective, Optimizer& opt)
        {

            if((std::is_same<Optimizer, lantern::perceptron::optimizer::StochasticGradientDescentWithMomentum>::value) ||
               (std::is_same<Optimizer,lantern::perceptron::optimizer::RootMeanSquarePropagation>::value) ||
               (std::is_same<Optimizer,lantern::perceptron::optimizer::AdaptiveMomentEstimation>::value)){

                #ifdef OPTIMIZE_VERSION
                if(!objective.IsVectorVelocityInit()){
                    objective.vector_velocity = std::move(lantern::utility::Vector<double>(max(objective.total_gradient_size + (objective.op == Activation::NOTHING? 0 : (objective.total_gradient_size == 0? 2 : 1)),1),0.0f));
                    objective.SetVectorVelocityInit(true);
                }

                for (auto &parent : objective.parents)
                {
                    if (!parent->IsVectorVelocityInit())
                    {
                        parent->vector_velocity = std::move(lantern::utility::Vector<double>(max(parent->total_gradient_size + (parent->op == Activation::NOTHING? 0 : (parent->total_gradient_size == 0? 2 : 1)),1),0.0f));
                        parent->SetVectorVelocityInit(true);
                    }
                }
                #endif
            }

            if((std::is_same<Optimizer, lantern::perceptron::optimizer::AdaptiveGradientDescent>::value) || 
               (std::is_same<Optimizer,lantern::perceptron::optimizer::AdaptiveMomentEstimation>::value)){
                
                #ifdef OPTIMIZE_VERSION

                if(!objective.IsPrevParamsInit()){
                    objective.stack_prev_gradient = std::move(lantern::utility::Vector<double>(max(objective.total_gradient_size + (objective.op == Activation::NOTHING? 0 : (objective.total_gradient_size == 0? 2 : 1)), 1), 0.0f));
                    objective.SetPrevParamsInit(true);
                }

                for (auto &parent : objective.parents)
                {
                    if (!parent->IsPrevParamsInit())
                    {
                        parent->stack_prev_gradient = std::move(lantern::utility::Vector(max(parent->total_gradient_size + (parent->op == Activation::NOTHING? 0 : (parent->total_gradient_size == 0? 2 : 1)), 1), 0.0));
                        parent->SetPrevParamsInit(true);
                    }
                }

                #endif
            }
            

            Perceptron *parent = nullptr;
            uint32_t index = 0;
            double gradient_of_function = 0;

            switch (objective.op)
            {
            case Activation::NATURAL_LOG:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::dlog(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::dlog(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif
            case Activation::EXP:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::dexp(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::dexp(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif
            case Activation::SIN:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::dsin(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::dsin(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif
            case Activation::COS:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::dcos(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::dcos(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif
            case Activation::TAN:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::dtan(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::dtan(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif
                break;
            case Activation::SIGMOID:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::dsigmoid(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::dsigmoid(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif 
                break;
            case Activation::RELU:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::drelu(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::drelu(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif 
                break; 
            case Activation::SWISH:
                #ifdef OPTIMIZE_VERSION
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input[objective.prev_child_index] * lantern::math::drelu(objective.value);
                    parent->gradient[index] -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input[index] = objective.gradient_based_input[objective.prev_child_index] * parent->gradient[index] * lantern::math::drelu(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient[parent->total_gradient_size] -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient[1] -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                #endif
                break;
            };
        };


        /**
         * @brief Backpropagate to optimize weight and bias, and use Gradient Descent as default optimizer
         * 
         * @tparam Optimizer 
         * @param first_node 
         * @param opt 
         */
        template <typename Optimizer>
        void BackPropagation(Perceptron &first_node, Optimizer& opt)
        {
            if (first_node.parents.empty())
            {
                return;
            }

            utility::Vector<Perceptron *> all_parents = {&first_node};

            uint32_t i = 0;
            Perceptron *current_node = nullptr;
            while (!all_parents.empty())
            {
                current_node = all_parents.back();
                CalculateGradient(*(current_node), opt);
                all_parents.pop_back();
                
                i = 0;
                for(;i < current_node->parents.size(); i++){
                    if(current_node->parents[i] != nullptr){
                        all_parents.push_back(current_node->parents[i]);
                    }
                }
            }
        }
        #endif

        #ifdef MATRIX_OPTIMIZE

        af::array multiply_elements(const af::array& lhs,const af::array& rhs){
            return lhs * rhs;
        }

        template <typename Optimizer>
        void BackPropagation(
            lantern::utility::Vector<af::array>& parameters,
            lantern::utility::Vector<af::array>& gradient_based_parameters,
            lantern::utility::Vector<lantern::perceptron::Activation>& operators,
            lantern::utility::Vector<af::array>& outputs,
            Optimizer& opt
        ){

            af::array gradient,gradient_weight, gradient_bias, all_gradient;

            for(int32_t i = parameters.size(); i > 0; i--){
                
                af::array& parameter = parameters[i];
                af::array& output = outputs[i];
                af::array& gradient_base_parameter = gradient_based_parameters[i + 1];
                lantern::perceptron::Activation& op = operators[i - 1];

                switch (op)
                {
                case Activation::SIGMOID:

                    // get all gradient from output
                    gradient = (output * (1 - output));
                    // the bias gradient is gradient from output multiply
                    // by one
                    gradient_bias = gradient;
                    // then create a vector of gradient with size of row is 
                    // "gradient row" multiply with "previous outputs row"
                    gradient = af::moddims(
                        af::tile(gradient,outputs[i-1].dims(0),1),
                        gradient.dims(0),
                        outputs[i-1].dims(0)
                    );
                    // transform "previous outputs" then multiply 
                    // each row of outputs with gradient then save into 
                    // gradient weight
                    gradient_weight = af::batchFunc(
                        outputs[i-1].T(),
                        gradient,
                        static_cast<af::batchFunc_t>(multiply_elements)
                    );
                    // join gradient_weight and gradient_bias
                    // to create a full gradient of weight and bias
                    all_gradient = af::join(0,gradient_weight.T(),gradient_bias.T());
                    // multiply all gradient of weight and bias with 
                    // prev weight and bias which multiply by weight not inputs
                    all_gradient = af::batchFunc(
                        gradient_base_parameter.T(),
                        all_gradient,
                        static_cast<af::batchFunc_t>(multiply_elements)
                    );

                    // update parameter with all gradient weight and bias
                    // and set optimizer GetDelta
                    parameter -= opt.GetDelta(all_gradient);
                    // last set based parameter to pass the current weight gradient
                    // to next 
                    gradient_based_parameters[i] = af::batchFunc(
                        gradient_base_parameter.T(),
                        parameter(af::seq(0,outputs[i-1].dims(0) - 1),af::span),
                        static_cast<af::batchFunc_t>(multiply_elements)
                    );

                    break;
                case Activation::RELU:

                    break;
                case Activation::SWISH:
                    
                    // get all gradient from output
                    gradient = (1/(1+af::exp(-output))) + ((output * (1 - output)) * output);
                    // the bias gradient is gradient from output multiply
                    // by one
                    gradient_bias = gradient;
                    // then create a vector of gradient with size of row is 
                    // "gradient row" multiply with "previous outputs row"
                    gradient = af::moddims(
                        af::tile(gradient,outputs[i-1].dims(0),1),
                        gradient.dims(0),
                        outputs[i-1].dims(0)
                    );
                    // transform "previous outputs" then multiply 
                    // each row of outputs with gradient then save into 
                    // gradient weight
                    gradient_weight = af::batchFunc(
                        outputs[i-1].T(),
                        gradient,
                        static_cast<af::batchFunc_t>(multiply_elements)
                    );
                    // join gradient_weight and gradient_bias
                    // to create a full gradient of weight and bias
                    all_gradient = af::join(0,gradient_weight.T(),gradient_bias.T());
                    // multiply all gradient of weight and bias with 
                    // prev weight and bias which multiply by weight not inputs
                    all_gradient = af::batchFunc(
                        gradient_base_parameter.T(),
                        all_gradient,
                        static_cast<af::batchFunc_t>(multiply_elements)
                    );

                    // update parameter with all gradient weight and bias
                    // and set optimizer GetDelta
                    parameter -= opt.GetDelta(all_gradient);
                    // last set based parameter to pass the current weight gradient
                    // to next 
                    gradient_based_parameters[i] = af::batchFunc(
                        gradient_base_parameter.T(),
                        parameter(af::seq(0,outputs[i-1].dims(0) - 1),af::span),
                        static_cast<af::batchFunc_t>(multiply_elements)
                    );

                    break;
                }

            }

        }
        #endif
        
    }
}

