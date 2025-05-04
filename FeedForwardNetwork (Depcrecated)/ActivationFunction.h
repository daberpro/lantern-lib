#pragma once
#include "Perceptron.h"
namespace lantern
{
    namespace perceptron
    {
        namespace activation
        {

            /**
             * @brief Natural Log Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void NaturalLog(Perceptron &n, Args &...args)
            {
                n.op = Activation::NATURAL_LOG;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief Exp Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void Exp(Perceptron &n, Args &...args)
            {
                n.op = Activation::EXP;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief Sin Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void Sin(Perceptron &n, Args &...args)
            {
                n.op = Activation::SIN;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief Cos Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void Cos(Perceptron &n, Args &...args)
            {
                n.op = Activation::COS;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief Tan Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void Tan(Perceptron &n, Args &...args)
            {
                n.op = Activation::TAN;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief Sigmoid Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void Sigmoid(Perceptron &n, Args &...args)
            {
                n.op = Activation::SIGMOID;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief ReLU Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void ReLU(Perceptron &n, Args &...args)
            {
                n.op = Activation::RELU;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief Siwsh Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void Swish(Perceptron &n, Args &...args)
            {
                n.op = Activation::SWISH;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief SoftMax Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void SoftMax(Perceptron &n, Args &...args)
            {
                n.op = Activation::SOFTMAX;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            /**
             * @brief Linear Activation Function
             * 
             * @tparam Args 
             * @param n 
             * @param args 
             */
            template <typename... Args>
            void Linear(Perceptron &n, Args &...args)
            {
                n.op = Activation::LINEAR;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

        }
    }
}