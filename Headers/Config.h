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
        static void to_json(json& j, const af::dim4& d) {
            j = json::array({ d[0], d[1], d[2], d[3] });
        }

        static void from_json(const json& j, af::dim4& d) {
            d = af::dim4(j.at(0), j.at(1), j.at(2), j.at(3));
        }
    };

    template <typename T>
    struct adl_serializer<lantern::utility::Vector<T>> {
        static void to_json(nlohmann::json& j, const lantern::utility::Vector<T>& vec) {
            j = nlohmann::json::array();
            for (size_t i = 0; i < vec.size(); ++i) {
                j.push_back(vec[i]);
            }
        }

        static auto from_json(const nlohmann::json& j) {
            lantern::utility::Vector<T> vec(j.size());
            for (size_t i = 0; i < j.size(); ++i) {
                vec.push_back(j.at(i).get<T>());
            }

            return vec;
        }
    };
}
