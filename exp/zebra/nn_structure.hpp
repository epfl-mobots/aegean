#include <simple_nn/loss.hpp>
#include <simple_nn/neural_net.hpp>

#include <iostream>

namespace nn_strucutre {
    void init_nn(std::vector<std::shared_ptr<simple_nn::NeuralNet>>& net, uint num_behaviours, uint num_individuals)
    {
        for (uint b = 0; b < num_behaviours; ++b) {
            net[b] = std::make_shared<simple_nn::NeuralNet>();
            net[b]->add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(25, 20);
            net[b]->add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(20, 20);
            net[b]->add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(20, 2); // dr cosdphi sindphi
            // std::cout << "NN " << b << " number of weights: " << net[b]->num_weights() << std::endl;
        }
    }

    void init_nn(simple_nn::NeuralNet& net, uint num_individuals)
    {
        net.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(25, 20);
        net.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(20, 20);
        net.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Tanh>>(20, 2); // dr cosdphi sindphi
    }

} // namespace nn_strucutre