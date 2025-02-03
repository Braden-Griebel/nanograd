//
// Created by bgriebel on 2/3/25.
//
#include "engine.h"

std::string Value::as_string() const {
    return "Value(data=" + std::to_string(this->val->data) +
           ", grad=" + std::to_string(this->val->grad) + ")";
}
