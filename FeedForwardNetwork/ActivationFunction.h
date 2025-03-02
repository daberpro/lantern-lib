#pragma once
#include "Perceptron.h"
namespace latern
{
    namespace perceptron
    {
        namespace activation
        {

            template <typename... Args>
            void NaturalLog(Perceptron &n, Args &...args)
            {
                n.op = Activation::NATURAL_LOG;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            template <typename... Args>
            void Exp(Perceptron &n, Args &...args)
            {
                n.op = Activation::EXP;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            template <typename... Args>
            void Sin(Perceptron &n, Args &...args)
            {
                n.op = Activation::SIN;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            template <typename... Args>
            void Cos(Perceptron &n, Args &...args)
            {
                n.op = Activation::COS;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            template <typename... Args>
            void Tan(Perceptron &n, Args &...args)
            {
                n.op = Activation::TAN;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            template <typename... Args>
            void Sigmoid(Perceptron &n, Args &...args)
            {
                n.op = Activation::SIGMOID;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            template <typename... Args>
            void ReLU(Perceptron &n, Args &...args)
            {
                n.op = Activation::RELU;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

            template <typename... Args>
            void Swish(Perceptron &n, Args &...args)
            {
                n.op = Activation::SWISH;
                ((n.child_index.push_back(args.total_gradient_size++), n.parents.push_back(&args)), ...);
            };

        }
    }
}