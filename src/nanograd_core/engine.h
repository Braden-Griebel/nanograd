#pragma once
// Standard Library Includes
#include <cmath>
#include <functional>
#include <ostream>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_set>
#include <vector>

// Local Includes

// External Includes

/**
 * @brief Represents a single scalar value and its gradient
 */
class InternalValue {
  /**
   * @brief The internal data associated with the Value.
   */
  double data;
  /**
   * @brief The value of the current derivative of the Value.
   *
   */
  double grad;
  /**
   * @brief Lambda expression used for calculating the
   * gradient during backpropagation.
   */
  std::function<void()> backwardsInternal;
  /**
   * @brief Children of the current value node.
   *
   */
  std::unordered_set<std::shared_ptr<InternalValue> > children;
  /**
   * @brief Operation that produced this node.
   *
   */
  std::string operation;

public:
  /**
   * @brief Construct a new Internal Value object
   *
   * @param data Internal data associated with the new value.
   * @param grad Current derivative value.
   * @param children Children of the new value node.
   * @param backwardsInternal Lambda expression for calculating the gradient
   *     of the new Value.
   * @param operation Operation which produced this node.
   */
  InternalValue(double data, double grad,
                std::unordered_set<std::shared_ptr<InternalValue> > children,
                std::function<void()> backwardsInternal,
                std::string operation)
    : data(data), grad(grad), children(children),
      backwardsInternal(backwardsInternal), operation(operation) {
  };
  /**
   * @brief Create a new InternalObject from a literal float value.
   *
   * @param data Data of the new Value being created.
   * @return std::shared_ptr<InternalValue>
   */
  static std::shared_ptr<InternalValue> valFromFloat(double data) {
    return std::make_shared<InternalValue>(InternalValue{
      data, 0., std::unordered_set<std::shared_ptr<InternalValue> >{},
      []() {
      },
      std::string{}
    });
  };

  /**
   * @brief Get the current value of the gradient.
   * @return grad, the current value of the gradient.
   */
  double get_grad() const {
    return this->grad;
  }

  /**
   * @brief Set the value of grad for the InternalValue.
   * @param grad New value for grad
   */
  void set_grad(const double grad) {
    this->grad = grad;
  }

  /**
   * @brief Get the current value of data
   * @return Current value of data
   */
  double get_data() const {
    return this->data;
  }

  /**
   * @brief Set the internal data value.
   * @param data New value for data
   */
  void set_data(double data) {
    this->data = data;
  }

  /**
   * @brief Set the value of grad to 0.0.
   */
  void zero_grad() {
    this->data = 0.0;
  }

  friend class Value;
};

class Value {
  /**
   * @brief Reference to the internal value.
   */
  std::shared_ptr<InternalValue> val;

  /**
   * @brief Topologically sort the expression graph starting from a given root.
   *
   * @param root Root Value to start the topological sort from.
   * @return std::vector<InternalValue> Nodes in topological order
   */
  static std::vector<std::shared_ptr<InternalValue> >
  topoSort(const Value *root) {
    // The nodes in topological order
    std::vector<std::shared_ptr<InternalValue> > topo{};
    // Nodes that have already been visited
    std::unordered_set<std::shared_ptr<InternalValue> > visited{};
    // Get the starting InternalValue
    std::shared_ptr<InternalValue> start = root->val;

    // Define lamba which will build the topological ordering
    std::function<void(const std::shared_ptr<InternalValue>)> build_topo;
    build_topo =
        [&](const std::shared_ptr<InternalValue> currentValue) -> void {
          if (!visited.count(currentValue)) {
            visited.insert(currentValue);
            for (const auto nextVal: currentValue->children) {
              build_topo(nextVal);
            }
            topo.push_back(currentValue);
          }
        };

    build_topo(start);

    return topo;
  }

public:
  /**
   * @brief Construct a new Value object.
   *
   * @param val Internal value held by the Value object.
   */
  Value(std::shared_ptr<InternalValue> val) : val(val) {
  };
  /**
   * @brief Construct a new Value object from a float literal.
   *
   * @param literalValue Literal float value from which to construct the new
   * Value.
   */
  Value(const double literalValue)
    : val(std::shared_ptr<InternalValue>{
      InternalValue::valFromFloat(literalValue)
    }) {
  };
  // region Access
  /**
   * @brief Get the current value of the gradient
   * @return Current value of the gradient
   */
  double get_grad() const {
    return this->val->get_grad();
  }

  /**
   * @brief Set the value of the gradient
   * @param grad New value of the gradient
   */
  void set_grad(const double grad) const {
    this->val->set_grad(grad);
  }

  /**
   * @brief Get the current value of data
   * @return data Current value of data
   */
  double get_data() const {
    return this->val->get_data();
  }

  /**
   * @brief Set the value of data.
   * @param data New value for underlying data value.
   */
  void set_data(const double data) const {
    this->val->set_data(data);
  }

  /**
   * @brief Set the value of grad to 0.
   */
  void zero_grad() const {
    this->val->zero_grad();
  }

  // endregion Access
  // region Operators

  /**
   * @brief Add two values.
   *
   * @param lhs Value on the left hand side of the addition
   * @param rhs Value on the right hand side of the addition
   * @return Value representing the two previous values being added
   */
  friend Value operator+(const Value &lhs, const Value &rhs) {
    // Create a new internal value for the addition node
    auto resInternalValue = std::make_shared<InternalValue>(
      lhs.val->data + rhs.val->data, // data
      0., // grad
      std::unordered_set<std::shared_ptr<InternalValue> >{
        lhs.val,
        rhs.val
      }, // children
      []() {
      }, // backwards lambda (defined later, since it needs a
      // reference to out)
      std::string{"+"} // operation
    );

    // Construct the Value to be returned
    Value out = Value{resInternalValue};

    // Construct the backwards function for the Out Value
    out.val->backwardsInternal = [&]() -> void {
      // Get references to the internal values
      std::shared_ptr<InternalValue> lhsInt = lhs.val;
      std::shared_ptr<InternalValue> rhsInt = rhs.val;
      std::shared_ptr<InternalValue> outInt = out.val;

      // Compute the gradients
      lhsInt->grad += outInt->grad;
      rhsInt->grad += outInt->grad;
    };

    return out;
  }

  friend Value operator+(const Value &lhs, double rhs) {
    return lhs + Value{rhs};
  }

  friend Value operator+(double lhs, const Value &rhs) {
    return Value{lhs} + rhs;
  }

  /**
   * @brief Multiply two values.
   *
   * @param lhs Value on the left hand side of the multiplication
   * @param rhs Value on the right hand side of the multiplication
   * @return Value representing the two previous values being multiplied

   */
  friend Value operator*(const Value &lhs, const Value &rhs) {
    auto resInternalValue = std::make_shared<InternalValue>(
      lhs.val->data * rhs.val->data, 0.,
      std::unordered_set<std::shared_ptr<InternalValue> >{lhs.val, rhs.val},
      []() {
      }, std::string{"*"});

    Value out = Value{resInternalValue};

    out.val->backwardsInternal = [&]() -> void {
      // Get references to the internal values
      std::shared_ptr<InternalValue> lhsInt = lhs.val;
      std::shared_ptr<InternalValue> rhsInt = rhs.val;
      std::shared_ptr<InternalValue> outInt = out.val;

      // Compute the gradients
      lhs.val->grad = rhs.val->data * out.val->grad;
      rhs.val->grad = lhs.val->data * out.val->grad;
    };

    return out;
  }

  friend Value operator*(const Value &lhs, double rhs) {
    return lhs * Value{rhs};
  }

  friend Value operator*(double lhs, const Value &rhs) {
    return Value{lhs} * rhs;
  }

  /**
   * @brief Raise a Value to an exponent.
   *
   * @param other Double representing the exponent
   * @return Value representing the previous value raised to the power of other
   */
  Value pow(double other) const {
    auto resInternalValue = std::make_shared<InternalValue>(
      std::pow(this->val->data, other), 0.,
      std::unordered_set<std::shared_ptr<InternalValue> >{this->val},
      []() {
      }, std::string{"**" + std::to_string(other)});

    Value out{resInternalValue};

    out.val->backwardsInternal = [&, other, this]() -> void {
      // Get references to the internal values
      const std::shared_ptr<InternalValue> baseInt = this->val;
      const std::shared_ptr<InternalValue> outInt = out.val;
      const double exponent = other;

      baseInt->grad +=
          (exponent * std::pow(baseInt->data, exponent - 1.0)) * outInt->grad;
    };

    return out;
  }

  /**
   * @brief Calculate a Rectified Linear Unit (ReLU) applied to the Value.
   *
   * @return Value representing Value after passing through the ReLU operation
   */
  Value relu() const {
    auto resInternalValue = std::make_shared<InternalValue>(
      this->val->data < 0. ? 0. : this->val->data, 0.,
      std::unordered_set<std::shared_ptr<InternalValue> >{this->val},
      []() {
      }, std::string{"ReLU"});

    Value out{resInternalValue};

    out.val->backwardsInternal = [&]() -> void {
      // Get references to internal values
      std::shared_ptr<InternalValue> selfInt = this->val;
      std::shared_ptr<InternalValue> outInt = out.val;

      selfInt->grad += (outInt->data > 0. ? outInt->grad : 0.);
    };

    return out;
  }

  /**
   * @brief Calculate the negative of a Value.
   *
   * @param val Value being multiplied by -1.
   * @return Value
   */
  friend Value operator-(const Value &val) {
    return val * -1.;
  };

  friend Value operator-(const Value &lhs, const Value &rhs) {
    return lhs + (-rhs);
  }

  friend Value operator-(const Value &lhs, double rhs) {
    return lhs - Value{rhs};
  }

  friend Value operator-(double lhs, const Value &rhs) {
    return Value{lhs} - rhs;
  }

  friend Value operator/(const Value &lhs, const Value &rhs) {
    return lhs * rhs.pow(-1.0);
  }

  friend Value operator/(const Value &lhs, double rhs) {
    return lhs / Value{rhs};
  }

  friend Value operator/(double lhs, const Value &rhs) {
    return Value{lhs} / rhs;
  }

  friend std::ostream &operator<<(std::ostream &os, const Value &val) {
    os << "Value(data=" << val.val->data << ", grad=" << val.val->grad << ")";
    return os;
  }

  /**
   * @brief Get a string representation of the Value.
   * @return String representing the Value
   */
  std::string as_string() const {
    return ("Value(data=" + std::to_string(this->val->data) +
            ", grad=" + std::to_string(this->val->grad) + ")");
  }


  // endregion Operators

  // region backpropogation

  /**
   * @brief Compute the Value of the gradients for the current Value
   */
  void backwards() const {
    // Start by topologically sorting the InternalValues
    auto nodes = Value::topoSort(this);

    /* Set value of this node to be 1 (since it is what
      the gradient is being calculated for)*/
    this->val->grad = 1.0;
    // Create a reverse view of the nodes vector

    // Iterate through the nodes in reverse order
    for (const std::ranges::reverse_view reverseNodes{nodes}; const std::shared_ptr<InternalValue> v: reverseNodes) {
      (v->backwardsInternal)();
    }
  }

  // endregion backpropogation
};
