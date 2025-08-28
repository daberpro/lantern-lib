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
                T value_{};
                auto [ptr_, e_] = std::from_chars(_str.data(), _str.data() + _str.size(), value_);
                if (e_ != std::errc{}) {
                    throw std::runtime_error(std::format("Error Utility ConvertFromString, ivalid conversion from value \"{}\" to type {}", _str, typeid(T).name()));
                }
                return value_;
            }
            else {
                if constexpr (std::is_same_v<T, std::string>) {
                    return _str;
                }
                else {
                    T value_{};
                    std::istringstream iss_(_str);
                    iss_ >> value_;
                    if (iss_.fail() || !iss_.eof()) {
                        throw std::runtime_error("Error Utility ConvertFromString,conversion failed or extra characters found.");
                    }
                    return value_;
                }
            }
        }

        /**
        * @brief Convert string into T type
        * @tparam T
        * @param _str
        * @return T
        */
        template <typename T>
        inline T ConvertFromString(const std::string_view& _str) {
            if constexpr (std::is_arithmetic_v<T>) {
                T value_{};
                auto [ptr_, e_] = std::from_chars(_str.data(), _str.data() + _str.size(), value_);
                if (e_ != std::errc{}) {
                    throw std::runtime_error(std::format("Error Utility ConvertFromString, ivalid conversion from value \"{}\" to type {}", _str, typeid(T).name()));
                }
                return value_;
            }
            else {
                if constexpr (std::is_same_v<T, std::string>) {
                    return _str;
                }
                else {
                    T value_{};
                    std::istringstream iss_(_str);
                    iss_ >> value_;
                    if (iss_.fail() || !iss_.eof()) {
                        throw std::runtime_error("Error Utility ConvertFromString,conversion failed or extra characters found.");
                    }
                    return value_;
                }
            }
        }

	}

}