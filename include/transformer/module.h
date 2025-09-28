#pragma once
#include <vector>
#include <memory>
#include "variable.h"

class Module {
    protected:
        std::vector<std::shared_ptr<Variable>> parameters;

        void registerParameter(const std::shared_ptr<Variable>& param) {
            parameters.push_back(param);
        }
    public:
        virtual ~Module() = default;

        virtual std::vector<std::shared_ptr<Variable>> getParameters() const {
            return parameters;
        }
};