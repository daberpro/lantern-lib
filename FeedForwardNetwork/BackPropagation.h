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

namespace latern {
    namespace perceptron {

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

            if((std::is_same<Optimizer, latern::perceptron::optimizer::StochasticGradientDescentWithMomentum>::value) ||
               (std::is_same<Optimizer,latern::perceptron::optimizer::RootMeanSquarePropagation>::value) ||
               (std::is_same<Optimizer,latern::perceptron::optimizer::AdaptiveMomentEstimation>::value)){

                if(!objective.IsVectorVelocityInit()){
                    objective.vector_velocity = af::constant(0.0,max(objective.total_gradient_size + (objective.op == Activation::NOTHING? 0 : (objective.total_gradient_size == 0? 2 : 1)), 1), f64);
                    objective.SetVectorVelocityInit(true);
                }

                for (auto &parent : objective.parents)
                {
                    if (!parent->IsVectorVelocityInit())
                    {
                        parent->vector_velocity = af::constant(0.0,max(parent->total_gradient_size + (parent->op == Activation::NOTHING? 0 : (objective.total_gradient_size == 0? 2 : 1)), 1), f64);
                        parent->SetVectorVelocityInit(true);
                    }
                }
            }

            if((std::is_same<Optimizer, latern::perceptron::optimizer::AdaptiveGradientDescent>::value) || 
               (std::is_same<Optimizer,latern::perceptron::optimizer::AdaptiveMomentEstimation>::value)){
                
                if(!objective.IsPrevParamsInit()){
                    objective.stack_prev_gradient = af::constant(0.0,max(objective.total_gradient_size + (objective.op == Activation::NOTHING? 0 : (objective.total_gradient_size == 0? 2 : 1)), 1), f64);
                    objective.SetPrevParamsInit(true);
                }

                for (auto &parent : objective.parents)
                {
                    if (!parent->IsPrevParamsInit())
                    {
                        parent->stack_prev_gradient = af::constant(0.0,max(parent->total_gradient_size + (parent->op == Activation::NOTHING? 0 : (objective.total_gradient_size == 0? 2 : 1)), 1), f64);
                        parent->SetPrevParamsInit(true);
                    }
                }
            }
            

            Perceptron *parent = nullptr;
            uint32_t index = 0;
            double gradient_of_function = 0;

            switch (objective.op)
            {
            case Activation::NATURAL_LOG:
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::dlog(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::dlog(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
            case Activation::EXP:
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::dexp(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::dexp(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
            case Activation::SIN:
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::dsin(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::dsin(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
            case Activation::COS:
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::dcos(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::dcos(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
            case Activation::TAN:
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::dtan(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::dtan(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };
                break;
            case Activation::SIGMOID:
                
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::dsigmoid(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::dsigmoid(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };

                break;
            case Activation::RELU:
                
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::drelu(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::drelu(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };

                break; 
            case Activation::SWISH:
                
                for (uint32_t i = 0; i < objective.parents.size(); i++)
                {
                    parent = objective.parents[i];
                    index = objective.child_index[i];
                    parent->prev_child_index = index;
                    gradient_of_function = objective.gradient_based_input(objective.prev_child_index, 0).scalar<double>() * latern::math::drelu(objective.value);
                    parent->gradient(index, 0) -= opt.GetDelta(gradient_of_function * parent->value, parent, index);
                    parent->gradient_based_input(index, 0) = objective.gradient_based_input(objective.prev_child_index, 0) * parent->gradient(index, 0) * latern::math::drelu(objective.value);
                    /**
                     * update for bias for parent and for objective itself because the bias always be the last gradient element
                     * so we just need to update gradient of parent one more time
                     * and check if parent was an input skip update bias for it, because 
                     * input no need bias
                     * 
                     * the objective only update it's own bias if the objective was output
                     */
                    if(parent->op != Activation::NOTHING){
                        parent->gradient(parent->total_gradient_size, 0) -= opt.GetDelta(gradient_of_function, parent, parent->total_gradient_size);
                    }
                    if(objective.total_gradient_size == 0){
                        objective.gradient(1,0) -= opt.GetDelta(gradient_of_function,&objective,1);
                    }
                };

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

        
    }
}

