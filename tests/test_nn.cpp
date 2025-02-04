// Standard Library Includes
#include <cmath>

// External Includes
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

// Local Includes
#include "nn.h"

TEST_CASE("Creating Modules", "[nn]") {
    SECTION("Creating Neurons") {
        Neuron a{5, true};
        // Margin for floating point
        double margin = 0.0000001;
        // Check that there are 6 parameters (5 weights and 1 bias)
        CHECK(a.get_parameters().size() == 6);
        const auto parameters = a.get_parameters();
        for (int idx = 0; idx < parameters.size()-1; idx++) {
            // For all the weights, check that they are not 0.
            CHECK(std::abs(parameters[idx].get_data())>margin);
        }
        // Check that the bias is near 0.0
        CHECK_THAT(parameters[parameters.size()-1].get_data(), Catch::Matchers::WithinAbs(0.0, margin));
    }
}