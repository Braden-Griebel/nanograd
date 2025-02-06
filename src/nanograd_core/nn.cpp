//
// Created by bgriebel on 2/3/25.
//
#include <nn.h>

#include <utility>

Neuron::Neuron(const int nin, const bool nonlinear): b(Value{0.}), nonlinear(nonlinear) {
    // create a random number generator
    std::random_device rd;
    std::mt19937 rng{rd()};
    std::uniform_real_distribution distribution(-1.0, 1.0);

    for (int i = 0; i < nin; i++) {
        double newWeight = distribution(rng);
        this->w.emplace_back(newWeight);
    }
}

Value Neuron::operator()(const std::vector<Value> &x) const {
    if (x.size() != this->w.size()) {
        throw std::runtime_error(
            "Neuron::operator(): mismatched size, w is of size " + std::to_string(this->w.size()) +
            " and x is of size " + std::to_string(x.size()));
    }
    Value activation = this->b;
    for (int idx = 0; idx < this->w.size(); ++idx) {
        activation = activation + (x[idx] * this->w[idx]);
    }
    if (this->nonlinear) {
        activation = activation.relu();
    }

    return activation;
}

std::vector<Value> Neuron::get_parameters() {
    std::vector<Value> out = this->w;
    out.push_back(this->b);
    return out;
}

auto Layer::get_parameters() -> std::vector<Value> {
    std::deque<Value> outDeque;
    for (auto& n: this->neurons) {
        auto neuronParams = n.get_parameters();
        outDeque.insert(outDeque.end(), neuronParams.begin(), neuronParams.end());
    }
    std::vector<Value> out{outDeque.begin(), outDeque.end()};
    return out;
}

std::vector<Value> Layer::operator()(const std::vector<Value> &x) const {
    std::vector<Value> out;
    for (const auto &neuron: neurons) {
        out.push_back(neuron(x));
    }
    return out;
}

std::vector<Value> MultiLayerPerceptron::operator()(std::vector<Value> x) const {
    std::vector<Value> out = std::move(x);
    for (auto& l : this->layers) {
        out = l(out);
    }
    return out;
}

std::vector<Value> MultiLayerPerceptron::get_parameters() {
    std::deque<Value> outDeque;
    for (auto& l: this->layers) {
        auto layerParams = l.get_parameters();
        outDeque.insert(outDeque.end(), layerParams.begin(), layerParams.end());
    }
    std::vector<Value> out{outDeque.begin(), outDeque.end()};
    return out;
}
