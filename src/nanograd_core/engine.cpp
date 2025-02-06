#include "engine.h"

InternalValue::InternalValue(const double data, const double grad,
    std::unordered_set<std::shared_ptr<InternalValue>> children, std::function<void()> backwardsInternal,
    std::string operation): data(data), grad(grad), backwardsInternal(std::move(backwardsInternal)),
                            children(std::move(children)), operation(std::move(operation)) {
}

std::shared_ptr<InternalValue> InternalValue::valFromFloat(double data) {
    return std::make_shared<InternalValue>(InternalValue{
        data, 0., std::unordered_set<std::shared_ptr<InternalValue> >{},
        []() {
        },
        std::string{}
    });
}

std::vector<std::shared_ptr<InternalValue>> Value::topoSort(const Value *root) {
    // The nodes in topological order
    std::vector<std::shared_ptr<InternalValue> > topo{};
    // Nodes that have already been visited
    std::unordered_set<std::shared_ptr<InternalValue> > visited{};
    // Get the starting InternalValue
    const std::shared_ptr<InternalValue> start = root->val;

    // Define lambda which will build the topological ordering
    std::function<void(const std::shared_ptr<InternalValue>)> build_topo;
    build_topo =
            [&](const std::shared_ptr<InternalValue> &currentValue) -> void {
                if (!visited.contains(currentValue)) {
                    visited.insert(currentValue);
                    for (const auto &nextVal: currentValue->children) {
                        build_topo(nextVal);
                    }
                    topo.push_back(currentValue);
                }
            };

    build_topo(start);

    return topo;
}

Value Value::pow(double other) const {
    const auto resInternalValue = std::make_shared<InternalValue>(
        std::pow(this->val->data, other), 0.,
        std::unordered_set<std::shared_ptr<InternalValue> >{this->val},
        []() {
        }, std::string{"**" + std::to_string(other)});

    Value out{resInternalValue};

    // Get references to the internal values
    const std::shared_ptr<InternalValue> baseInt = this->val;
    const std::shared_ptr<InternalValue> outInt = out.val;
    const double exponent = other;

    // Capture by value to get counted references to the internal values, without needing
    // to access the wrapping Value (which can then be managed more easily in python)
    out.val->backwardsInternal = [=]() -> void {
        baseInt->grad +=
                (exponent * std::pow(baseInt->data, exponent - 1.0)) * outInt->grad;
    };

    return out;
}

Value Value::relu() const {
    const auto resInternalValue = std::make_shared<InternalValue>(
        this->val->data < 0. ? 0. : this->val->data, 0.,
        std::unordered_set<std::shared_ptr<InternalValue> >{this->val},
        []() {
        }, std::string{"ReLU"});

    Value out{resInternalValue};

    // Get references to internal values
    const std::shared_ptr<InternalValue> selfInt = this->val;
    const std::shared_ptr<InternalValue> outInt = out.val;

    out.val->backwardsInternal = [=]() -> void {
        selfInt->grad += (outInt->data > 0. ? outInt->grad : 0.);
    };

    return out;
}

std::string Value::as_string() const {
    return "Value(data=" + std::to_string(this->val->data) +
           ", grad=" + std::to_string(this->val->grad) + ")";
}

auto Value::backwards() const -> void {
    // Start by topologically sorting the InternalValues
    auto nodes = Value::topoSort(this);

    /* Set value of this node to be 1 (since it is what
      the gradient is being calculated for)*/
    this->val->grad = 1.0;
    // Create a reverse view of the nodes vector

    // Iterate through the nodes in reverse order
    for (const std::ranges::reverse_view reverseNodes{nodes}; const std::shared_ptr<InternalValue> &v: reverseNodes) {
        (v->backwardsInternal)();
    }
}

std::ostream & operator<<(std::ostream &os, const Value &val) {
    os << "Value(data=" << val.val->data << ", grad=" << val.val->grad << ")";
    return os;
}
