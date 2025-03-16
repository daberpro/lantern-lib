#pragma once
#include "Perceptron.h"
#include "Layering.h"
#include "ActivationFunction.h"
#include "PerceptronFeedForward.h"
#include "BackPropagation.h"
#include "LossFunction.h"
#include "Loader.h"

namespace lantern {
    
    template <typename Optimizer>
    class FeedForwardNetwork{
    private:
        uint32_t layer = 0, batch_size = 3;
        std::atomic<uint32_t> epoch = 100;
        double max_treshold = 1e-04;

        Optimizer optimizer;

        lantern::perceptron::Layer model_layer;
        lantern::utility::Vector<lantern::perceptron::Perceptron> layer_inputs;
        lantern::utility::Vector<lantern::utility::Vector<lantern::perceptron::Perceptron>> layer_hiddens;
        lantern::utility::Vector<lantern::perceptron::Perceptron> layer_outputs;

        #ifdef MATRIX_OPTIMIZE
        lantern::utility::Vector<af::array> batch_gradient;
        lantern::utility::Vector<af::array> parameters;
    	lantern::utility::Vector<af::array> gradient_based_parameters;
        lantern::utility::Vector<af::array> outputs;
        lantern::utility::Vector<lantern::perceptron::Activation> operators;
        #endif

    public:

        void SetEpoch(const uint32_t& epoch) {
            this->epoch = epoch;
        }

        void SetBatchSize(const uint32_t& batch_size){
            this->batch_size = batch_size;
        }

        void SetMaxTreshold(const double& max_treshold){
            this->max_treshold = max_treshold;
        }

        void SetOptimizerFrom(Optimizer& opt){
            this->optimizer = opt;
        }

        template <lantern::perceptron::Activation activation>
        void AddInputLayer(uint32_t total_perceptron){
            if(this->layer == 0){
                for(uint32_t i = 0; i < total_perceptron; i++){
                    this->layer_inputs.push_back(
                        lantern::perceptron::Perceptron("i" + std::to_string(i))
                    );
                }
                return;
            }
            std::cerr << "Lantern Feed Forward Network Error, cannot add input layer after other type of layer\n";
            std::cerr << "input layer must be call first before other layer\n";
            exit(EXIT_FAILURE);
        };

        template <lantern::perceptron::Activation activation>
        void AddHiddenLayer(uint32_t total_perceptron){
            lantern::utility::Vector<lantern::perceptron::Perceptron*> parents;
            if(this->layer == 0){
                for(auto& p : this->layer_inputs){
                    parents.push_back(&p);
                }
            }else{
                lantern::utility::Vector<lantern::perceptron::Perceptron>& l = this->layer_hiddens[this->layer-1];
                for(auto& p : l){
                    parents.push_back(&p);
                }
            }
            
            this->layer++;
            lantern::utility::Vector<lantern::perceptron::Perceptron> temp;
            
            for(uint32_t i = 0; i < total_perceptron; i++){
                lantern::perceptron::Perceptron h("h"+std::to_string(this->layer)+"_"+std::to_string(i));
                h.layer = this->layer;
                h.op = activation;
                h.parents.copyPtrData(parents);
                temp.push_back(
                    std::move(h)
                );
            }

            this->layer_hiddens.push_back(std::move(temp));

        };

        template <lantern::perceptron::Activation activation>
        void AddOutputLayer(uint32_t total_perceptron){
            lantern::utility::Vector<lantern::perceptron::Perceptron*> parents;
            lantern::utility::Vector<lantern::perceptron::Perceptron>& l = this->layer_hiddens[this->layer-1];
            for(auto& p : l){
                parents.push_back(&p);
            }
            
            this->layer++;
            for(uint32_t i = 0; i < total_perceptron; i++){
                lantern::perceptron::Perceptron o("o"+std::to_string(this->layer)+"_"+std::to_string(i));
                o.layer = this->layer;
                o.op = activation;
                o.parents.copyPtrData(parents);
                this->layer_outputs.push_back(
                    std::move(o)
                );
            }
        };

        void InitModel(){
            uint32_t hidden_size = this->layer_hiddens.size();
            this->model_layer.SetLayer(this->layer_outputs,hidden_size + 1);
            for(int32_t i = hidden_size - 1; i >= 0; i--){
                this->model_layer.SetLayer(this->layer_hiddens[i],i+1);
            }
            this->model_layer.SetLayer(this->layer_inputs,0);

            #ifdef MATRIX_OPTIMIZE
            
            lantern::perceptron::FeedForward(
                this->model_layer, 
                this->parameters, 
                this->gradient_based_parameters, 
                this->operators, 
                this->outputs, 
                this->optimizer, 
                this->batch_gradient
            );

            this->gradient_based_parameters.push_back(af::constant(1.0f, 1, f64));
            #endif

        }

        void ShowParameters(){
            std::cout << std::string(70,'=') << '\n';
            std::cout << "Model parameters" << '\n';
            std::cout << std::string(70,'=') << '\n';
            
            uint32_t layer = 0;
            #ifdef MATRIX_OPTIMIZE
            for(auto& param: this->parameters){
                std::cout << af::toString(("Layer "+std::to_string(layer)).c_str(),param,16,true) << '\n';
                std::cout << std::string(70,'=') << '\n';
                layer++;
            }
            #endif
        }

        void Train(af::array& input_data, af::array& target_data){

            std::atomic<int32_t> iteration(0);
            std::atomic<double> loss(0);
            std::atomic<bool> stop(false);

            std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
            std::thread learning([&]()-> void {
                
                std::random_device rd;
                std::mt19937 rg(rd());
                std::uniform_int_distribution<> dis(0, input_data.dims(0)-1);
            
                #ifdef MATRIX_OPTIMIZE
                uint32_t i = 0, batch_iter = 0;
                af::array output;
                double loss_total = 0;

                while (iteration < this->epoch) {
                    i = dis(rg);
                    parameters[0] = input_data.row(i).T();
                    output = target_data.row(i);
                    
                    lantern::perceptron::FeedForward(
                        this->parameters, 
                        this->operators, 
                        this->outputs
                    );

                    loss = lantern::perceptron::loss::SumSquaredResidual(this->outputs.back(), output);
                    
                    if(this->batch_size == 1)
                    {
                        this->gradient_based_parameters.back() = lantern::perceptron::loss::DerivativeSumSquaredResidual(outputs.back(), output);
                        lantern::perceptron::BackPropagation(
                            this->parameters, 
                            this->gradient_based_parameters, 
                            this->operators, 
                            this->outputs, 
                            this->optimizer
                        );
                        iteration++;
                    }
                    else
                    {
                        if(batch_iter % this->batch_size == 0 && batch_iter != 0){
                            
                            /**
                             * calculate average gradient 
                             * and reset batch_iter to 0
                             * then update each parameter
                             * using optimizer 
                             */
                            for(int32_t p = parameters.size() - 1; p > 0; p--){
                                this->batch_gradient[p] /= this->batch_size;
                                this->batch_gradient[p].eval();
                                this->parameters[p] -= this->optimizer.GetDelta(this->batch_gradient[p],p);
                                this->parameters[p].eval();
                            }
                            batch_iter = 0;
                            iteration++;
        
                        }else{
                            this->gradient_based_parameters.back() = lantern::perceptron::loss::DerivativeSumSquaredResidual(this->outputs.back(), output);
                            lantern::perceptron::CalculateGradient(
                                this->parameters, 
                                this->gradient_based_parameters, 
                                this->operators, 
                                this->outputs, 
                                this->optimizer,
                                this->batch_gradient
                            );
                        }
                    }
        
                    // if(loss < this->max_treshold){
                    //     break;
                    // }

                    batch_iter++;
                }

                #endif
                stop = true;
            });

            std::thread progress([&]()-> void {
                while(!stop){
                    lantern::perceptron::ProgressBar(iteration,this->epoch);
                    std::cout << std::fixed << std::setw(5) << " | Loss : " << std::setw(16) << std::setprecision(16) << loss << ", Iteration : " << std::setw(5) << iteration;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    std::cout << std::flush;
                }
        
                if(stop && iteration < this->epoch){
                    lantern::perceptron::ProgressBar(this->epoch,this->epoch);
                    std::cout << std::fixed << std::setw(5) << " | Loss : " << std::setw(16) << std::setprecision(16) << loss << ", Iteration : " << std::setw(5) << iteration;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    std::cout << std::flush;
                }
            });

            learning.join();
            progress.join();

            std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> total_running_times = end - start;
            std::cout << "\n";
            std::cout << "Total time: " << total_running_times.count() << "s\n";

        };

        void Predict(af::array& input_data, lantern::utility::Vector<af::array>& result){
            for(uint32_t i = 0; i < input_data.dims(0); i++){
                this->parameters[0] = input_data.row(i).T();
                lantern::perceptron::FeedForward(
                    this->parameters, 
                    this->operators, 
                    this->outputs
                );

                result.push_back(this->outputs.back());
            }
        };

    };

}

