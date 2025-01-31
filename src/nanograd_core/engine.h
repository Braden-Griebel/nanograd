#ifndef ENGINE_H
#define ENGINE_H
// Standard Library Includes
#include <functional>
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
  std::optional<std::function<double()>> backwardsInternal;
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
   * @param children Children of the new value node.
   * @param backwardsInternal Lambda expression for calculating the gradient
   *     of the new Value.
   * @param operation Operation which produced this node.
   */
  InternalValue(double data,
                std::unordered_set<std::shared_ptr<InternalValue>> children,
                std::optional<std::function<int()>> backwardsInternal, std::string operation)
      : data(data), children(children), backwardsInternal(backwardsInternal),
        operation(operation) {

        };
};

class Value {
  std::shared_ptr<InternalValue> val;
};

#endif // ENGINE_H
