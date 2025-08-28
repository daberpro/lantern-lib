#pragma once
#include "../pch.h"
#include "Vector.h"

/**
 * ================================================================
 * This file contain all config for lantern such as serializer,
 * deserializer, and other things
 * 
 * ================================================================
 */

namespace nlohmann {
    template <>
    struct adl_serializer<af::dim4> {
        static void to_json(json& _j, const af::dim4& _d) {
            _j = json::array({ _d[0], _d[1], _d[2], _d[3] });
        }

        static void from_json(const json& _j, af::dim4& _d) {
            _d = af::dim4(_j.at(0), _j.at(1), _j.at(2), _j.at(3));
        }
    };

    template <typename T>
    struct adl_serializer<lantern::utility::Vector<T>> {
        static void to_json(nlohmann::json& _j, const lantern::utility::Vector<T>& _vec) {
            _j = nlohmann::json::array();
            for (size_t i = 0; i < _vec.size(); ++i) {
                _j.push_back(_vec[i]);
            }
        }

        static auto from_json(const nlohmann::json& _j) {
            lantern::utility::Vector<T> vec_(_j.size());
            for (size_t i = 0; i < _j.size(); ++i) {
                vec_.push_back(_j.at(i).get<T>());
            }

            return vec_;
        }
    };
}
