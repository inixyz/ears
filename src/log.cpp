#include "log.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>

std::string time() {
  auto const now = std::chrono::current_zone()->to_local(std::chrono::system_clock::now());

  return std::put_time("%F %T");
}

void message(const std::ostream &stream, ) {
}
