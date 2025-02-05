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

    SECTION("Creating Layers") {
        Layer testLayer{4,5,true};
        // Margin for floating point
        const double margin = 0.0000001;

        // Check that there are 25 parameters (4 weights 1 bias per each of 5 neurons)
        CHECK(testLayer.get_parameters().size() == 25);

        // Check that the layer can be called
        const std::vector<Value> inputs {Value{1.0}, Value{1.0}, Value{1.0}, Value{1.0}};
        const std::vector<Value> outputs = testLayer(inputs);

        // Again since the input is all 1.0, all the parameters should have gradients of 1.0
        for (auto& output: outputs) {
            // Additionally, all the outputs should be non-negative due to the ReLU activation
            CHECK(output.get_data() >=0.0);
            output.backwards();
        }
        for (auto& param: testLayer.get_parameters()) {
            CHECK_THAT(param.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));
        }

        // Zero the gradients and check that this worked
        testLayer.zero_grad();
        for (auto& param: testLayer.get_parameters()) {
            CHECK_THAT(param.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        }
    }

    SECTION("Creating MultiLayerPerceptrons") {
        std::vector layerSizes {5,5,4,3};
        MultiLayerPerceptron testMultiLayerPerceptron{4,layerSizes};
        // Margin for floating point
        const double margin = 0.0000001;

        /* The layers have:
         * -> 5 neurons * (4 weights + 1 bias) = 25 total
         * -> 5 neurons * (5 weights + 1 bias) = 30 total
         * -> 4 neurons * (5 weights + 1 bias) = 24 total
         * -> 3 neurons * (4 weights + 1 bias) = 15 total
         * Overall: 74 total parameters
         */
        CHECK(testMultiLayerPerceptron.get_parameters().size() == 94);

        // Check that the Layer can be called
        const std::vector<Value> inputs = {Value{1.0},Value{1.0},Value{1.0},Value{1.0},};
        const std::vector<Value> outputs = testMultiLayerPerceptron(inputs);

        // The size of the last layer is 3, so there should be 3 outputs
        CHECK(outputs.size() == 3);

        // Check that the gradients can be computed
        for (auto& out: outputs) {
            out.backwards();
        }
        for (auto& param: testMultiLayerPerceptron.get_parameters()) {
            // Due to the ReLU all the gradients should be non-negative
            CHECK(param.get_grad()>=0.0);
        }
        // Check that zeroing the gradients works
        testMultiLayerPerceptron.zero_grad();
        for (auto& param: testMultiLayerPerceptron.get_parameters()) {
            CHECK_THAT(param.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        }

    }
}