#include "sim/aegean_simulation.hpp"

#include <simple_nn/neural_net.hpp>

// #include <tools/polygons/circular_corridor.hpp>
// #include <tools/archive.hpp>
// #include <tools/mathtools.hpp>

// #include <tools/polygons/circular_corridor.hpp>
// #include <tools/reconstruction/cspace.hpp>

// #include <features/alignment.hpp>
// #include <features/inter_individual_distance.hpp>

// #include <Eigen/Core>
// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <iomanip>
// #include <cmath>

// using namespace aegean;
// using namespace tools;

// #define USE_POLAR

// struct Params {
//     struct CircularCorridor : public defaults::CircularCorridor {
//     };
// };

int main(int argc, char** argv)
{
    simple_nn::NeuralNet network;
    // add layers here

    simu::simulation::AegeanSimulation sim(network);
    sim.spin_once();

    return 0;
}