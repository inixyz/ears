#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "core/vec.cuh"
#include "core/world.cuh"
using namespace ears;

// clang-format off
PYBIND11_MODULE(ears, m) {
  py::class_<Vec3i>(m, "Vec3i")
    .def(py::init<const int &, const int &, const int &>())
    .def(py::init([](const py::tuple &tuple) {
      if (tuple.size() != 3)
        throw std::runtime_error("Tuple must have exactly 3 elements to be converted to Vec3i.");
      return Vec3i(tuple[0].cast<int>(), tuple[1].cast<int>(), tuple[2].cast<int>());
    }))
    .def_readwrite("x", &Vec3i::x)
    .def_readwrite("y", &Vec3i::y)
    .def_readwrite("z", &Vec3i::z);

  py::implicitly_convertible<py::tuple, Vec3i>();

  py::class_<World>(m, "World")
    .def(py::init<const Vec3i &, const float, const Vec3i &, const Vec3i &>())

    .def("get_size", &World::get_size)
    .def("get_courant", &World::get_courant)

    .def("get_t0", &World::get_t0)
    .def("get_t1", &World::get_t1)
    .def("get_t2", &World::get_t2)

    .def("set_t0", &World::set_t0)
    .def("set_t1", &World::set_t1)
    .def("set_t2", &World::set_t2)

    .def("step", py::overload_cast<>(&World::step))
    .def("step", py::overload_cast<const int>(&World::step));
}
// clang-format on
