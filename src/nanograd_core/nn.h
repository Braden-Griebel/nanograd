#pragma once

// Standard Library Imports
#include <utility>
#include <vector>
#include <deque>
#include <random>

// External Imports

// Local Imports
#include "engine.h"

/**
 * @brief Base class for all neural network associated objects
 */
class Module {
    std::vector<Value> params;

public:
    explicit Module(const std::vector<Value> &params) : params(params) {
    }

    Module() = default;

    virtual ~Module() = default;

    /**
     * @brief Zero the gradients of parameters associated with the module.
     */
    virtual void zero_grad() {
        for (const auto &param: params) {
            param.zero_grad();
        }
    }

    /**
     * @brief Get the parameters associated with the module
     * @return The parameters associated with the module
     */
    [[nodiscard]] virtual std::vector<Value> get_parameters() {
        return this->params;
    }
};


/**
 * @brief Neuron represents a single neuron in a neural network
 */
class Neuron final : public Module {
    /**
     * @brief The weights of the neuron
     */
    std::vector<Value> w;
    /**
     * @brief The bias of the neuron
     */
    Value b;
    /**
     * @brief Whether the output should be non-linear (via ReLU)
     */
    bool nonlinear;

public:
    /**
     * @brief Construct a neuron with nin inputs
     * @param nin Number of inputs to the neuron
     * @param nonlinear Whether the neuron should use a non-linear activation function (ReLU)
     */
    explicit Neuron(const int nin, const bool nonlinear);;

    /**
     * @brief Determine the activation of the neuron given an input
     * @param x Vector of values coming in to the neuron (must be the same length as w)
     * @return Neuron activation
     */
    Value operator()(const std::vector<Value> &x) const;;

    /**
     * @brief Get the parameters of the neurone (weights and bias)
     * @return Vector of parameters of the neuron
     */
    std::vector<Value> get_parameters() override;

    /**
     * @brief Zero the gradients of the parameters of the neuron
     */
    void zero_grad() override {
        for (auto &param: this->w) {
            param.zero_grad();
        }
        this->b.zero_grad();
    };
};

/**
 * @brief Represents a single layer of neurons in a neural network
 */
class Layer final : public Module {
    std::vector<Neuron> neurons;

public:
    /**
     * @brief Create a layer of randomly initialized neurons
     * @param nin Number of inputs to the layer
     * @param nout Number of outputs from the layer
     * @param nonlinear Whether the neurons should include a non-linear layer
     */
    Layer(const int nin, const int nout, const bool nonlinear) {
        for (int i = 0; i < nout; ++i) {
            neurons.emplace_back(nin, nonlinear);
        }
    }

    /**
     * @brief Find all the parameters of the Layer
     * @return Parameters of the Layer
     */
    std::vector<Value> get_parameters() override;;

    /**
     * @brief Calculate the neuron activations given an input x
     * @param x Input vector to this layer
     * @return Vector of neuron activations/outputs from this Layer
     */
    std::vector<Value> operator()(const std::vector<Value> &x) const;

    /**
     * @brief Zero the gradients of all the neurons in the Layers.
     */
    void zero_grad() override {
        for (auto &neuron: neurons) {
            neuron.zero_grad();
        }
    }
};


class MultiLayerPerceptron final : public Module {
    std::vector<Layer> layers;

public:
    /**
     * @brief Create a MultiLayerPerceptron
     * @param nin Number of inputs to the Multilayer Perceptron
     * @param nouts Vector of Layer sizes for the MultiLayerPerceptron
     */
    MultiLayerPerceptron(int nin, std::vector<int> nouts) {
        this->layers.emplace_back(nin, nouts[0], 0 != nouts.size() - 1);
        for (int idx = 0; idx < nouts.size()-1; ++idx) {
            this->layers.emplace_back(nouts[idx], nouts[idx + 1], idx != nouts.size() - 2);
        }
    }

    /**
     * @brief Run the MultiLayerPerceptron on a given input
     * @param x Input of vector of Values to the MultiLayerPerceptron
     * @return Activation values of the last layer of the MultiLayerPerceptron
     */
    std::vector<Value> operator()(std::vector<Value> x) const;

    /**
     * @brief Get all the parameters associated with the MultiLayerPerceptron
     * @return A vectors of the parameters for all Layers in the perceptron
     */
    std::vector<Value> get_parameters() override;

    /**
     * @brief Zero the gradients of all the Layers in the MultiLayerPerceptron
     */
    void zero_grad() override {
        for (auto &l: this->layers) {
            l.zero_grad();
        }
    }

};

