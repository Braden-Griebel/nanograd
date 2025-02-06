// Standard Library Includes
#include <iostream>

// External Includes
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

// Local Includes
#include "engine.h"

TEST_CASE("Calculating Gradients", "[engine]") {
    SECTION("Creating Values") {
        Value x{1.0};
        Value y{2.0};
        // Margin for floating point
        double margin = 0.0000001;
        // Check that the values are correctly created
        CHECK_THAT(x.get_data(), Catch::Matchers::WithinAbs(1.0, margin));
        CHECK_THAT(y.get_data(), Catch::Matchers::WithinAbs(2.0, margin));
        // Check that the gradients start at 0.0
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
    }

    SECTION("For Addition") {
        Value x{5.};
        Value y{3.};
        Value z = x+y;
        // Define the floating point margin
        double margin = 0.0000001;

        // Gradients should start at 0.
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(z.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        // Check that the values are correctly created
        CHECK_THAT(x.get_data(), Catch::Matchers::WithinAbs(5.0, margin));
        CHECK_THAT(y.get_data(), Catch::Matchers::WithinAbs(3.0, margin));
        CHECK_THAT(z.get_data(), Catch::Matchers::WithinAbs(8.0, margin));
        // Check the gradient for dz/dx and dz/dy
        z.backwards(); // caclulate the gradients
        // Check that the grad values for x and y are correct
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));
    }

    SECTION("For Multiplication") {
        Value x{5.};
        Value y{3.};
        Value z = x*y;
        // Define the floating point margin
        double margin = 0.0000001;

        // Gradients should start at 0.
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(z.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        // Check that the values are correctly created
        CHECK_THAT(x.get_data(), Catch::Matchers::WithinAbs(5.0, margin));
        CHECK_THAT(y.get_data(), Catch::Matchers::WithinAbs(3.0, margin));
        CHECK_THAT(z.get_data(), Catch::Matchers::WithinAbs(15.0, margin));
        // Check the gradient for dz/dx and dz/dy
        z.backwards(); // caclulate the gradients
        // Check that the grad values for x and y are correct
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(3.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(5.0, margin));
    }

    SECTION("For Exponentiation") {
        Value x{5.};
        Value y = x.pow(3.0);
        // Define the floating point margin
        double margin = 0.0000001;

        // Gradients should start at 0.
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        // Check that the values are correctly created
        CHECK_THAT(x.get_data(), Catch::Matchers::WithinAbs(5.0, margin));
        CHECK_THAT(y.get_data(), Catch::Matchers::WithinAbs(125.0, margin));

        // Check the gradient for dz/dx and dz/dy
        y.backwards(); // calculate the gradients
        // Check that the grad values for x and y are correct
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(75.0, margin)); // x^3->3*x^2->50
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));
    }

    SECTION("For ReLU") {
        Value x{5.};
        Value y = x.relu();

        // Define the floating point margin
        double margin = 0.0000001;

        // Gradients should start at 0.
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        // Check that the values are correctly created
        CHECK_THAT(x.get_data(), Catch::Matchers::WithinAbs(5.0, margin));
        CHECK_THAT(y.get_data(), Catch::Matchers::WithinAbs(5.0, margin));

        // Check the gradient for dz/dx and dz/dy
        y.backwards(); // calculate the gradients
        // Check that the grad values for x and y are correct
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(1.0, margin)); // x^3->2*x^2->50
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));

        // Testing with a negative value
        Value a{-2.0};
        Value b = a.relu();

        // Gradients should start at 0.
        CHECK_THAT(a.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(b.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        // Check that the values are correctly created
        CHECK_THAT(a.get_data(), Catch::Matchers::WithinAbs(-2.0, margin));
        CHECK_THAT(b.get_data(), Catch::Matchers::WithinAbs(0.0, margin));

        // Check the gradient for db/da
        b.backwards(); // calculate the gradients
        // Check that the grad values for a and b are correct
        CHECK_THAT(a.get_grad(), Catch::Matchers::WithinAbs(0.0, margin)); // x^3->2*x^2->50
        CHECK_THAT(b.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));
    }

    SECTION("Test Zeroing"){
        Value x{5.};
        Value y{3.};
        Value z = x+y;
        // Define the floating point margin
        double margin = 0.0000001;

        // Gradients should start at 0.
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(z.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        // Check the gradient for dz/dx and dz/dy
        z.backwards(); // caclulate the gradients
        // Check that the grad values for x and y are correct
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(1.0, margin));
        // Zero the gradients
        x.zero_grad();
        y.zero_grad();
        z.zero_grad();
        // Gradients should now be 0
        CHECK_THAT(x.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(y.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
        CHECK_THAT(z.get_grad(), Catch::Matchers::WithinAbs(0.0, margin));
    }

    SECTION("More Complex Calculation") {
        Value a {-4.0};
        Value b {2.0};
        Value c = a+b;
        // Next calculation
        Value d = a*b + b.pow(3.0);
        c = c + c + 1.0;
        c = c + 1.0 + c + (-a);
        d = d+d*2.0 + (b+a).relu();
        d = d+3.0*d+(b-a).relu();
        Value e=c-d;
        Value f=e.pow(2.0);
        Value g = f / 2.0;
        g = g + 10.0 / f;
        // Below tests based on micrograd
        CHECK_THAT(g.get_data(), Catch::Matchers::WithinAbs(24.7041, 0.0001));
        g.backwards();
        CHECK_THAT(a.get_grad(), Catch::Matchers::WithinAbs(138.8338, 0.0001));
        CHECK_THAT(b.get_grad(), Catch::Matchers::WithinAbs(645.5773, 0.0001));
    }
}
