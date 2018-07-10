#include "ElementsKernel/ProgramHeaders.h"

class GPz : public Elements::Program {
public :

    Elements::ExitCode mainMethod(std::map<std::string, variable_value>& args) override {
        return Elements::ExitCode::OK;
    }
};

MAIN_FOR(GPz)
