#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>

#include <iostream>

namespace nn_strucutre {
    static constexpr uint DIMS_IN = 24;
    static constexpr uint DIMS_OUT = 2;

    void init_nn(simple_nn::NeuralNet& net, uint num_individuals)
    {
        net.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(DIMS_IN, 30);
        net.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(30, 30);
        // net.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(30, 30);
        net.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(30, DIMS_OUT);
    }

    void init_nn(std::vector<std::shared_ptr<simple_nn::NeuralNet>>& net, uint num_behaviours, uint num_individuals)
    {
        for (uint b = 0; b < num_behaviours; ++b) {
            net[b] = std::make_shared<simple_nn::NeuralNet>();
            init_nn(*net[b], num_individuals);
            // std::cout << "NN " << b << " number of weights: " << net[b]->num_weights() << std::endl;
        }
    }
} // namespace nn_strucutre