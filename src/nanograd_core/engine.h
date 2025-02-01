#ifndef ENGINE_H
#define ENGINE_H
// Standard Library Includes
#include <functional>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

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
  std::optional<std::function<void()>> backwardsInternal;
  /**
   * @brief Children of the current value node.
   *
   */
  std::unordered_set<std::shared_ptr<InternalValue>> children;
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
                std::unordered_set<std::shared_ptr<InternalValue>> children,
                std::optional<std::function<void()>> backwardsInternal,
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
        data, 0., std::unordered_set<std::shared_ptr<InternalValue>>{},
        std::nullopt, std::string{}});
  };
  friend class Value;
};

class Value {
  /**
   * @brief Reference to the internal value.
   *
   */
  std::shared_ptr<InternalValue> val;

public:
  /**
   * @brief Construct a new Value object.
   *
   * @param val Internal value held by the Value object.
   */
  Value(std::shared_ptr<InternalValue> val) : val(val) {};
  /**
   * @brief Construct a new Value object from a float literal.
   *
   * @param literalValue Literal float value from which to construct the new
   * Value.
   */
  Value(double literalValue)
      : val(std::shared_ptr<InternalValue>{
            InternalValue::valFromFloat(literalValue)}) {};
  // region Operators

  /**
   * @brief Add two values.
   *
   * @param lhs Value on the left hand side of the addition
   * @param rhs Value on the right hand side of the addition
   * @return New Value representing the two previous values being added
   */
  friend Value operator+(const Value &lhs, const Value &rhs) {
    // Create a new internal value for the addition node
    InternalValue resInternalValue{
        lhs.val->data + rhs.val->data,                        // data
        0.,                                                   // grad
        std::unordered_set<std::shared_ptr<InternalValue>>{
          lhs.val, rhs.val
        }, // children
        std::nullopt,    // backwards lambda (defined later, since it needs a
                         // reference to out)
        std::string{"+"} // operation
    };

    // Construct the Value to be returned
    Value out = Value{std::make_shared<InternalValue>(&resInternalValue)};

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

  friend Value operator*(const Value& lhs, const Value& rhs){
    InternalValue resInternalValue {
      lhs.val->data * rhs.val->data,
      0., 
      std::unordered_set<std::shared_ptr<InternalValue>>{
        lhs.val, rhs.val
      }, 
      std::nullopt, 
      std::string{"*"}
    };

    Value out = Value{std::make_shared<InternalValue>(&resInternalValue)};

    out.val->backwardsInternal = [&]()->void {
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

  Value pow(double other){
    InternalValue resInternalValue {
      std::pow(this->val->data, other),
      0., 
      std::unordered_set<std::shared_ptr<InternalValue>>{
        this->val
      },
      std::nullopt,
      std::string{"**"+std::to_string(other)}
    };

    Value out = Value{std::make_shared<InternalValue>(&resInternalValue)};

    out.val->backwardsInternal = [&]()->void{
      this->val->grad += (other * std::pow(this->val->data, other-1.0))*out.val->grad;
    };

    return out;
  }

  Value relu(){
    InternalValue resInternalValue {
      this->val->data < 0. ? 0. : this->val->data,
       0., 
       std::unordered_set<std::shared_ptr<InternalValue>>{
        this->val
       },
       std::nullopt, 
       std::string{"ReLU"}
    };

    Value out {std::make_shared<InternalValue>(&resInternalValue)};

    out.val->backwardsInternal = [&]()->void{
      this->val->grad += (out.val->data > 0. ? out.val->grad : 0. );
    };

    return out;
  }

  // endregion Operators
};

#endif // ENGINE_H
