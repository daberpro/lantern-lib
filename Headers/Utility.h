#pragma once
#include "../pch.h"

namespace lantern {

	namespace utility {

        /**
        * @brief Convert string into T type
        * @tparam T
        * @param _str
        * @return T
        */
        template <typename T>
        inline T ConvertFromString(const std::string& _str) {
            if constexpr (std::is_arithmetic_v<T>) {
                T value{};
                auto [ptr, e] = std::from_chars(_str.data(), _str.data() + _str.size(), value);
                if (e != std::errc{}) {
                    throw std::runtime_error("Error Utility ConvertFromString, ivalid conversion");
                }
                return value;
            }
            else {
                if constexpr (std::is_same_v<T, std::string>) {
                    return _str;
                }
                else {
                    T value{};
                    std::istringstream iss(_str);
                    iss >> value;
                    if (iss.fail() || !iss.eof()) {
                        throw std::runtime_error("Error Utility ConvertFromString,conversion failed or extra characters found.");
                    }
                    return value;
                }
            }
        }

	}

}