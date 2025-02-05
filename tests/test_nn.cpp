// Standard Library Includes
#include <cmath>

// External Includes
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

// Local Includes
#include "nn.h"

TEST_CASE("Creating Modules", "[nn]") {
    SECTION("Creating Neurons") {
        Neuron testNeuron{5, true};
        // Margin for floating point
        double margin = 0.0000001;
        // Check that there are 6 parameters (5 weights and 1 bias)
        CHECK(testNeuron.get_parameters().size() == 6);
        const auto parameters = testNeuron.get_parameters();
        for (int idx = 0; idx < parameters.size()-1; idx++) {
            // For all the weights, check that they are not 0.
            CHECK(std::abs(parameters[idx].get_data())>margin);
        }
        // Check that the bias is near 0.0
        CHECK_THAT(parameters[parameters.size()-1].get_data(), Catch::Matchers::WithinAbs(0.0, margin));

        // Check that the neuron can be called
        const std::vector<Value> inputs = {Value{1.0},Value{1.0},Value{1.0},Value{1.0},Value{1.0}};
        const Value output = testNeuron(inputs);

        // Since the output is fed through relu, it should be non-negative
        CHECK(output.get_data() > margin);

        // Check that the gradients can be calculated
        output.backwards();

        // The gradients should all be non-zero
        for (auto& param : testNeuron.get_parameters()) {
            CHECK(param.get_grad()> margin);
        }

        // Zero the gradients
        testNeuron.zero_grad();

        // Check that all the gradients are now 0.
        for (auto& param: testNeuron.get_parameters()) {
            CHECK_THAT(param.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        }
    }
}