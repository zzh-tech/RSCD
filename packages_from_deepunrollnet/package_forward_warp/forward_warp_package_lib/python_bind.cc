#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "flow_forward_shift.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Flow_forward_shift>(m, "Flow_forward_shift")
            .def(py::init<at::Tensor, int, int, float>())
            .def("forward", &Flow_forward_shift::forward)
            .def("backward", &Flow_forward_shift::backward);
}



