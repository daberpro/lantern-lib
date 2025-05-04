#pragma once
#include "../pch.h"
#include <matplot/matplot.h>
#include "Perceptron.h"
#include "Layering.h"
#include "ActivationFunction.h"
#include "PerceptronFeedForward.h"
#include "BackPropagation.h"
#include "LossFunction.h"
#include "Loader.h"
#include "../Headers/Logging.h"

namespace lantern {
    
    template <typename Optimizer>
    class FeedForwardNetwork{
    private:
        uint32_t layer = 0, batch_size = 3;
        std::atomic<uint32_t> epoch = 100;
        double max_treshold = 1e-07;

        Optimizer optimizer;

        lantern::perceptron::Layer model_layer;
        lantern::utility::Vector<lantern::perceptron::Perceptron> layer_inputs;
        lantern::utility::Vector<lantern::utility::Vector<lantern::perceptron::Perceptron>> layer_hiddens;
        lantern::utility::Vector<lantern::perceptron::Perceptron> layer_outputs;

        
        lantern::utility::Vector<af::array> batch_gradient;
        lantern::utility::Vector<af::array> parameters;
    	lantern::utility::Vector<af::array> gradient_based_parameters;
        lantern::utility::Vector<af::array> outputs;
        lantern::utility::Vector<lantern::perceptron::Activation> operators;
        
        lantern::utility::Vector<af::array> batch_inputs;
        lantern::utility::Vector<af::array> batch_targets;
        lantern::utility::Vector<uint32_t> size_of_class_data;
        lantern::utility::Vector<uint32_t> batch_input_data_labels;
        
        uint32_t class_size = 0;
        uint32_t rest_of_each_class_data = 0;
        uint32_t size_will_put_of_each_class = 1;
        uint32_t total_data_on_batch = 0;

        /**
         * @brief Batching data for input and output
         * 
         * @param input_data 
         * @param target_data 
         */
        void Batching(af::array& input_data, af::array& target_data){
            
            /**
             * Clear all batch inputs and outputs 
             * to save memory
             */
            this->batch_inputs.clear();
            this->batch_targets.clear();
            this->batch_input_data_labels.clear();

            /**
             * find how much each class data need to feed 
             * on every batch
             */
            
            std::random_device rd;
            std::mt19937 rg(rd());
            std::uniform_int_distribution<> dis(0,100);
            
            uint32_t prev_size_of_class_data = 0;
            uint32_t index_of_random_class_data = 0;

            for(uint32_t i = 0; i < this->class_size; i++){
                
                if(this->class_size == 1){
                    dis = std::uniform_int_distribution<>(
                        prev_size_of_class_data, 
                        input_data.dims(0) - 1
                    );
                }
                else
                {
                    dis = std::uniform_int_distribution<>(
                        prev_size_of_class_data, 
                        prev_size_of_class_data + this->size_of_class_data[i] - 1
                    );
                }

                for(uint32_t j = 0; j < this->size_will_put_of_each_class; j++){
                    index_of_random_class_data = dis(rg);
                    this->batch_input_data_labels.push_back(i);
                    this->batch_inputs.push_back(input_data.row(index_of_random_class_data).T());
                    this->batch_targets.push_back(target_data.row(index_of_random_class_data).T());
                }

                prev_size_of_class_data += this->size_of_class_data[i];

            }

            /**
             * Get rest of data if the model was classification model
             * if not skip this because this will create redundant of input data
             */

             if(this->class_size != 1){

                 /**
                  * Get rest of data to make the batch data full
                  */
                 prev_size_of_class_data = 0;
                 index_of_random_class_data = 0;
                 uint32_t index_of_class_data = 0;
                 for(uint32_t j = 0; j < this->rest_of_each_class_data; j++){
                     
                     if(index_of_random_class_data > this->class_size){
                         prev_size_of_class_data = 0;
                         index_of_class_data = 0;
                     }
                     
                     this->batch_input_data_labels.push_back(index_of_class_data);
                     dis = std::uniform_int_distribution<>(
                         prev_size_of_class_data, 
                         prev_size_of_class_data + this->size_of_class_data[index_of_class_data]
                     );
     
                     index_of_random_class_data = dis(rg);
                     this->batch_inputs.push_back(input_data.row(index_of_random_class_data).T());
                     this->batch_targets.push_back(target_data.row(index_of_random_class_data).T());
                     prev_size_of_class_data += this->size_of_class_data[index_of_class_data];
                     index_of_class_data++;
                 }
             }

        }

    public:

        /**
         * @brief Get the Parameters of model
         * 
         * @return lantern::utility::Vector<af::array> 
         */
        lantern::utility::Vector<af::array> GetParameters(){
            return this->parameters;
        }

        /**
         * @brief Get the Outputs of model
         * 
         * @return lantern::utility::Vector<af::array> 
         */
        lantern::utility::Vector<af::array> GetOutputs(){
            return this->outputs;
        }

        /**
         * @brief Set the Size Of Class Data object
         * 
         * @tparam Args 
         * @param args 
         */
        template <typename... Args>
        void SetSizeOfClassData(Args... args){
            (
                this->size_of_class_data.push_back(args),
                ...
            );
        }

        /**
         * @brief Set the Epoch 
         * 
         * @param epoch 
         */
        void SetEpoch(const uint32_t& epoch) {
            this->epoch = epoch;
        }

        /**
         * @brief Set the Batch Size
         * 
         * @param batch_size 
         */
        void SetBatchSize(const uint32_t& batch_size){
            this->batch_size = batch_size;
        }

        /**
         * @brief Set the Max Treshold, when the value of loss function was below then this,training will be stop
         * 
         * @param max_treshold 
         */
        void SetMaxTreshold(const double& max_treshold){
            this->max_treshold = max_treshold;
        }

        /**
         * @brief Set the Optimizer From outside definiton
         * 
         * @param opt 
         */
        void SetOptimizerFrom(Optimizer& opt){
            this->optimizer = opt;
        }

        /**
         * @brief Add input layer to network
         * 
         * @tparam activation 
         * @param total_perceptron 
         */
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

        /**
         * @brief Add Hidden layer to network
         * 
         * @tparam activation 
         * @param total_perceptron 
         */
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

        /**
         * @brief Add Ouput layer to network
         * 
         * @tparam activation 
         * @param total_perceptron 
         */
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

        void PrintDetail(){
            this->model_layer.PrintTotalNodeOnLayer();
        }

        /**
         * @brief initalize model parameters
         * 
         */
        void InitModel(){
            
            uint32_t hidden_size = this->layer_hiddens.size();
            this->model_layer.SetLayer(this->layer_outputs,hidden_size + 1);
            for(int32_t i = hidden_size - 1; i >= 0; i--){
                this->model_layer.SetLayer(this->layer_hiddens[i],i+1);
            }
            this->model_layer.SetLayer(this->layer_inputs,0);

            this->class_size = this->size_of_class_data.size();
            // preventing to mod by 0
            if(this->class_size == 0){
                this->class_size = 1;
            }
            /**
             * get rest of class data need to put using
             * rest = batch_size % class_size
             * size_to_put = (batch_size - rest) / class_size
             */
            this->rest_of_each_class_data = this->batch_size % this->class_size;
            this->total_data_on_batch = (this->batch_size - this->rest_of_each_class_data);
            this->size_will_put_of_each_class = this->total_data_on_batch / this->class_size;

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
            

        }

        /**
         * @brief Show parameters of model such as weights and bias
         * 
         */
        void ShowParameters(){
            std::cout << std::string(70,'=') << '\n';
            std::cout << "Model parameters" << '\n';
            std::cout << std::string(70,'=') << '\n';
            
            uint32_t layer = 0;
            
            for(auto& param: this->parameters){
                std::cout << af::toString(("Layer "+std::to_string(layer)).c_str(),param,16,true) << '\n';
                std::cout << std::string(70,'=') << '\n';
                layer++;
            }
            
        }

        /**
         * @brief Train model using input and output data, and loss function and its derivative
         * 
         * @tparam LossFunctionType 
         * @tparam DLossFunctionType 
         * @param input_data 
         * @param target_data 
         * @param LossFunction 
         * @param DerivativeLossFunction 
         */
        template <typename LossFunctionType = double,typename DLossFunctionType = af::array>
        void Train(af::array& input_data, af::array& target_data, std::function<LossFunctionType(af::array&,af::array&)> LossFunction, std::function<DLossFunctionType(af::array&,af::array&)> DerivativeLossFunction){

            std::atomic<int32_t> iteration(0);
            std::atomic<double> loss(0);
            std::atomic<bool> stop(false);

            std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
            std::thread learning([&]()-> void {
                
                uint32_t i = 0, j = 0;
                af::array output;
                double loss_total = 1;

                /**
                 * this is just use for classification model
                 * eventually for softmax
                 */
                lantern::utility::Vector<af::array> outputs_probability_of_class;
                af::array output_probability_of_class;
                af::array current_output;
                af::array current_gradient_of_softmax;

                while (iteration < this->epoch) {

                    /**
                     * Get input and output for batching 
                     */
                    this->Batching(input_data,target_data);

                    for(; i < this->batch_inputs.size(); i++){

                        this->parameters[0] = this->batch_inputs[i];
                        output = this->batch_targets[i];
                        
                        lantern::perceptron::FeedForward(
                            this->parameters, 
                            this->operators, 
                            this->outputs
                        );
                        
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
                            loss_total = loss;
                        }
                        else
                        {

                            /**
                             * check if the model was a classification model
                             * then i need to take all outputs from each class
                             * to calculate it for softmax
                             */
                            if(this->class_size > 1){

                                current_output = this->outputs.back();
                                
                                // if((i + 1) < this->total_data_on_batch && ((i + 1) % this->size_will_put_of_each_class != 0 || i == 1)){
                                    if(output_probability_of_class.isempty()){
                                        output_probability_of_class = current_output;
                                    }else{
                                        output_probability_of_class += current_output;
                                        output_probability_of_class.eval();
                                    }
                                // }else if((i + 1) % this->size_will_put_of_each_class == 0){
                                //     outputs_probability_of_class.push_back(
                                //         output_probability_of_class 
                                //     );
                                //     /**
                                //      * reset the outputs to next class outputs
                                //      */
                                //     output_probability_of_class = af::array();
                                // }else{

                                //     /**
                                //      * take the rest of batch outputs
                                //      */
                                //     if(j >= this->class_size){
                                //         j = 0;
                                //     }
                                //     outputs_probability_of_class[j] += current_output;
                                //     outputs_probability_of_class[j].eval();
                                //     j++;

                                // }



                            }else{

                                /**
                                 * This is mini-batch computation
                                 * this will work if the number of batch was set
                                 * more than 0
                                 * 
                                 * in here we do batch normalization and other operation
                                 * which need mini-batch
                                 */
      
                                this->gradient_based_parameters.back() = DerivativeLossFunction(this->outputs.back(), output);
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

                    }

                    if(this->class_size > 1){
                        
                        // output_probability_of_class = af::array();
                        // for(auto out: outputs_probability_of_class){
                            
                        //     if(output_probability_of_class.isempty()){
                        //         output_probability_of_class = out / this->size_will_put_of_each_class;
                        //         output_probability_of_class.eval();
                        //     }else{
                        //         output_probability_of_class = af::join(
                        //             1,
                        //             output_probability_of_class,
                        //             out / this->size_will_put_of_each_class
                        //         );
                        //     }
                        //     i++;
                        // }
                        
                        // output_probability_of_class -= af::diag(
                        //     af::constant(1.0f, output_probability_of_class.dims(0), f64),
                        //     0,
                        //     false
                        // );
                        // output_probability_of_class.eval();
                        
                        // output_probability_of_class = af::sum(output_probability_of_class,1);
                        
                        // TODO: Update this, see on TODO.txt
                        for(uint32_t p = 0; p < this->total_data_on_batch; p++){
                            
                            this->parameters[0] = this->batch_inputs[p];
                            
                            lantern::perceptron::FeedForward(
                                this->parameters, 
                                this->operators, 
                                this->outputs
                            );

                            current_gradient_of_softmax = output_probability_of_class - this->batch_targets[p];
                            // current_gradient_of_softmax = output_probability_of_class.row(this->batch_input_data_labels[p]).T();
                            this->gradient_based_parameters.back() = DerivativeLossFunction(
                                // output_probability_of_class, 
                                current_gradient_of_softmax, 
                                output
                            );

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
                
                    /**
                     * calculate average gradient 
                     * and reset batch_iter to 0
                     * then update each parameter
                     * using optimizer 
                     */
                    for(int32_t p = this->parameters.size() - 1; p > 0; p--){
                        this->batch_gradient[p] /= this->batch_size;
                        this->batch_gradient[p].eval();
                        this->parameters[p] -= this->optimizer.GetDelta(this->batch_gradient[p],p);
                        this->parameters[p].eval();
                    }

                    output_probability_of_class = af::array();
                    // to save memory
                    outputs_probability_of_class.clear();
                    
                    iteration++;
                    i = 0;
                    loss = LossFunction(this->outputs.back(), output);
                    loss_total = loss;
                    std::cout << "Loss : " << loss << '\n';

                }

                
                stop = true;
            });

            // std::thread progress([&]()-> void {
            //     while(!stop){
            //         lantern::perceptron::ProgressBar(iteration,this->epoch);
            //         std::cout << std::fixed << std::setw(5) << " | Loss : " << std::setw(16) << std::setprecision(16) << loss << ", Iteration : " << std::setw(5) << iteration;
            //         std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //         std::cout << std::flush;
            //     }
        
            //     if(stop && iteration < this->epoch){
            //         lantern::perceptron::ProgressBar(this->epoch,this->epoch);
            //         std::cout << std::fixed << std::setw(5) << " | Loss : " << std::setw(16) << std::setprecision(16) << loss << ", Iteration : " << std::setw(5) << iteration;
            //         std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //         std::cout << std::flush;
            //     }
            // });

            learning.join();
            // progress.join();

            std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> total_running_times = end - start;
            std::cout << "\n";
            std::cout << "Total time: " << total_running_times.count() << "s\n";

        };

        /**
         * @brief Get prediciton result after training model
         * 
         * @param input_data 
         * @param result 
         */
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

        /**
         * @brief Show a graph of network
         * 
         */
        void ShowGraph(){
            // ! Draw graph of model 
        }

    };

}

